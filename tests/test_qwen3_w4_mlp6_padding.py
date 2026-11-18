"""Packed W4 G32 MLP6 zero padding on the host: math, loaders and admission.

Qwen3 4B/14B/32B and Qwen3-VL 4B/32B keep MLP6 at a six-core budget by
zero-padding the physical MLP width to a multiple of 32*6 once, before the
single pgrp pack. These checks decode the actual physical payload on CPU; they
are not RPU kernel or board evidence.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rpu_backend.api._execution import qwen3_vl_text_topology
from rpu_backend.quant import convert_qwen3_w4a16 as w4
from rpu_backend.quant import load_qwen3_vl_awq as awq
from rpu_backend.quant import load_qwen3_vl_runtime as runtime
from rpu_backend.quant.int4_pgrp_pack import pad_int4_mlp_stripe, w4_storage_shape
from rpu_backend.runtime.topology import (
    decoder_mlp_intermediate_size, decoder_packed_w4_mlp_intermediate_size,
)
from test_qwen3_vl_awq_packing import exact_config, inverse_rpu_scale, inverse_rpu_weight, source_pack


def _dequantize(linear, partition, cores):
    """Physical pgrp payload -> logical FP64 weight [N,K] of its Linear."""
    n, k = linear.out_features, linear.in_features
    values = inverse_rpu_weight(linear.weight, n, k, partition, cores)
    scales = inverse_rpu_scale(linear.weight_scale, n, k, partition, cores)
    return values.double() * scales.double().repeat_interleave(32, dim=1)


def _mlp(x, gate, up, down):
    return (torch.nn.functional.silu(x @ gate.t()) * (x @ up.t())) @ down.t()


@pytest.mark.parametrize("logical,cores,expected", [
    (9728, 6, 9792), (17408, 6, 17472), (25600, 6, 25728),
    (3072, 6, 3072), (6144, 6, 6144), (12288, 6, 12288),
    (9728, 4, 9728), (17408, 8, 17408), (25600, 4, 25600),
])
def test_packed_width_is_192_aligned_only_for_six_lanes(logical, cores, expected):
    physical = decoder_packed_w4_mlp_intermediate_size(logical, cores)
    assert physical == expected
    assert physical % (32 * cores) == 0  # per-core Down K stripe is whole 32-K tiles
    # FP16/W8 keep their existing 96-aligned six-lane widths.
    assert decoder_mlp_intermediate_size(9728, 6) == 9792
    assert decoder_mlp_intermediate_size(17408, 6) == 17472


@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_qwen3_vl_packed_topology_is_the_shared_template(size, cores):
    text = exact_config(size).text_config
    topology = qwen3_vl_text_topology(text, cores, packed_w4=True)
    assert topology == qwen3_vl_text_topology(text, cores)
    assert (topology.attn_tp, topology.mlp_tp, topology.lm_head_tp) == (
        (8, 8, 8) if cores == 8 else (4, cores, cores))


@pytest.mark.parametrize("group_size", [32, None])
def test_padded_mlp6_math_equals_unpadded_mlp4_on_cpu(group_size):
    """Zero rows/columns of the padded stripe contribute nothing to the MLP."""
    torch.manual_seed(7)
    hidden, logical = 128, 512  # 512 % 192 != 0: six lanes need 576
    physical = decoder_packed_w4_mlp_intermediate_size(logical, 6)
    assert physical == 576
    source = {name: (torch.randn(shape) / 8).half() for name, shape in (
        ("gate", (logical, hidden)), ("up", (logical, hidden)), ("down", (hidden, logical)))}
    packed = {}
    for cores in (4, 6):
        layer = {}
        for name, partition in (("gate", 1), ("up", 1), ("down", 0)):
            linear = nn.Linear(*reversed(source[name].shape), bias=False, dtype=torch.float16)
            linear.weight = nn.Parameter(source[name].clone(), requires_grad=False)
            role = f"mlp.{name}_proj"
            w4.pack_linear_int4_(linear, partition, name=role, group_size=group_size,
                                 num_cores=cores,
                                 padded_size=w4.packed_mlp_padded_size(role, linear, cores))
            layer[name] = (linear, partition)
        packed[cores] = layer
    four, six = packed[4], packed[6]
    assert (four["gate"][0].out_features, four["down"][0].in_features) == (logical, logical)
    assert (six["gate"][0].out_features, six["down"][0].in_features) == (physical, physical)
    assert six["down"][0].weight.shape == (hidden, physical // 2)
    weights = {}
    for cores, layer in packed.items():
        weights[cores] = {name: _dequantize(linear, partition, cores)
                          for name, (linear, partition) in layer.items()}
    # Logical quantization is unchanged; padded rows/columns are exact zero.
    for name, axis in (("gate", 0), ("up", 0), ("down", 1)):
        padded = weights[6][name]
        logical_part = padded[:logical] if axis == 0 else padded[:, :logical]
        tail = padded[logical:] if axis == 0 else padded[:, logical:]
        assert torch.equal(logical_part, weights[4][name])
        assert torch.count_nonzero(tail) == 0
    # Padded scales are unit, so no denormal/NaN scale can reach the kernel.
    down_scale = inverse_rpu_scale(six["down"][0].weight_scale, hidden, physical, 0, 6)
    assert torch.equal(down_scale[:, logical // 32:], torch.ones_like(down_scale[:, logical // 32:]))
    x = torch.randn(5, hidden, dtype=torch.float64)
    activation = (torch.nn.functional.silu(x @ weights[6]["gate"].t())
                  * (x @ weights[6]["up"].t()))
    assert torch.count_nonzero(activation[:, logical:]) == 0
    reference = _mlp(x, weights[4]["gate"], weights[4]["up"], weights[4]["down"])
    torch.testing.assert_close(_mlp(x, weights[6]["gate"], weights[6]["up"], weights[6]["down"]),
                               reference, rtol=0, atol=1e-12)


@pytest.mark.parametrize("partition,n,k,padded", [(1, 512, 128, 576), (0, 128, 512, 576)])
def test_awq_and_runtime_pack_pad_values_and_scales_identically(partition, n, k, padded):
    q = ((torch.arange(n * k).reshape(n, k) * 5 + torch.arange(n)[:, None] * 3) % 16 - 8).to(torch.int8)
    scales = (torch.arange(n * (k // 32)).reshape(n, -1) % 47 + 1).to(torch.bfloat16) / 256
    weight, scale = awq._pack_projection(source_pack(q), torch.tensor([n, k]), scales,
                                         n=n, k=k, partition=partition, num_cores=6,
                                         padded_size=padded)
    pn, pk = (padded, k) if partition == 1 else (n, padded)
    assert weight.shape == (pn, pk // 2)
    values = inverse_rpu_weight(weight, pn, pk, partition, 6)
    physical_scales = inverse_rpu_scale(scale, pn, pk, partition, 6)
    assert torch.equal(values[:n, :k], q) and torch.count_nonzero(values) == torch.count_nonzero(q)
    assert torch.equal(physical_scales[:n, :k // 32], scales.half())
    tail = physical_scales[n:] if partition == 1 else physical_scales[:, k // 32:]
    assert torch.equal(tail, torch.ones_like(tail))
    unsigned = q.to(torch.int16) & 15
    raw = (unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)).to(torch.uint8)
    runtime_weight, runtime_scale = runtime._pack_w4_projection(
        raw, scales.half().t().contiguous(), n=n, k=k, partition=partition,
        num_cores=6, padded_size=padded)
    assert torch.equal(runtime_weight, weight) and torch.equal(runtime_scale, scale)


def test_padding_rejects_shrink_split_groups_and_wrong_scale_shape():
    values, scales = torch.zeros(128, 512, dtype=torch.int8), torch.ones(16, 128, dtype=torch.float16)
    for kwargs in (dict(partition=0, padded_size=500), dict(partition=0, padded_size=528),
                   dict(partition=1, padded_size=64), dict(partition=2, padded_size=768)):
        with pytest.raises(ValueError):
            pad_int4_mlp_stripe(values, scales, **kwargs)
    with pytest.raises(ValueError):
        pad_int4_mlp_stripe(values, scales[:, :64], partition=0, padded_size=768)


def _tiny_qwen3(hidden=512, intermediate=512):
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.self_attn, layer.mlp = nn.Module(), nn.Module()
        for owner, name, n, k in (("self_attn", "q_proj", hidden, hidden),
                                  ("self_attn", "k_proj", 128, hidden),
                                  ("self_attn", "v_proj", 128, hidden),
                                  ("self_attn", "o_proj", hidden, hidden),
                                  ("mlp", "gate_proj", intermediate, hidden),
                                  ("mlp", "up_proj", intermediate, hidden),
                                  ("mlp", "down_proj", hidden, intermediate)):
            linear = nn.Linear(k, n, bias=False, dtype=torch.float16)
            linear.weight.data.normal_(0, 0.05)
            getattr(layer, owner).add_module(name, linear)
        model.model.layers.append(layer)
    return model


@pytest.mark.parametrize("mlp_tp,width", [(6, 576), (4, 512), (8, 512)])
def test_converter_pads_only_the_six_lane_mlp_and_keeps_attention(mlp_tp, width):
    model = _tiny_qwen3()
    attn_tp = 8 if mlp_tp == 8 else 4
    topology = SimpleNamespace(attn_tp=attn_tp, mlp_tp=mlp_tp, lm_head_tp=attn_tp)
    w4.convert_qwen3_to_w4a16_(model, group_size=32, topology=topology)
    for layer in model.model.layers:
        for role, partition in w4._PROJECTIONS:
            linear = layer.get_submodule(role)
            n, k = linear.out_features, linear.in_features
            cores = mlp_tp if role.startswith("mlp.") else attn_tp
            assert linear.weight.dtype == torch.uint8
            assert linear.weight.shape == w4_storage_shape(n, k, partition, cores)
            if mlp_tp == 8 and role in ("self_attn.k_proj", "self_attn.v_proj"):
                # Sixteen logical output rows per core occupy one N32 block.
                # The Linear keeps its logical width while each core's tail is zero.
                assert n == 128 and linear.weight.shape == (256, 256)
                padded = inverse_rpu_weight(linear.weight, 256, k, partition, cores)
                assert torch.count_nonzero(padded.reshape(8, 32, k)[:, 16:]) == 0
            if role in ("mlp.gate_proj", "mlp.up_proj"):
                assert (n, k) == (width, 512)
            elif role == "mlp.down_proj":
                assert (n, k) == (512, width)
            else:
                assert (n, k) == {"self_attn.k_proj": (128, 512),
                                  "self_attn.v_proj": (128, 512)}.get(role, (512, 512))
            local_g, local_n = (k // 32, n // cores) if partition else (k // 32 // cores, n)
            count = cores * ((local_g + 3) // 4 * 4) * ((local_n + 63) // 64 * 64)
            assert linear.weight_scale.shape == (32, count // 32)


def test_tiny_awq_loader_pads_six_lane_mlp_and_records_physical_width(tmp_path, monkeypatch):
    from test_qwen3_vl_runtime_quant_loading import TinyStorage, write_tiny_awq_source

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        cfg, values = write_tiny_awq_source(tmp_path / "awq", size=4)
        monkeypatch.setattr(awq.AutoModelForImageTextToText, "from_config", TinyStorage)
        topology = qwen3_vl_text_topology(cfg.text_config, 6, packed_w4=True)
        assert (topology.attn_tp, topology.mlp_tp, topology.lm_head_tp) == (4, 6, 6)
        model = awq.load_qwen3_vl_awq_imagetext(
            cfg, str(tmp_path / "awq"), local_files_only=True, topology=topology)
    finally:
        torch.set_num_threads(previous)
    assert awq._validate_qwen3_vl_awq_model(model)
    assert model.config.text_config.intermediate_size == 9728  # config stays logical
    for name, module, partition in awq._projection_inventory(model)[:7]:
        n, k = values[name + ".weight_shape"].tolist()
        cores = topology.mlp_tp if ".mlp." in name else topology.attn_tp
        pn, pk = awq.packed_projection_geometry(name, n, k, partition, topology.mlp_tp)
        assert (module.out_features, module.in_features) == (pn, pk)
        assert (module._rpu_linear_partition, module._rpu_linear_num_cores) == (partition, cores)
        if ".mlp." in name:  # tiny storage: 128 rows -> 384, 512 columns -> 576
            assert (pn, pk) == ((192, 512) if partition else (128, 576))
        decoded = inverse_rpu_weight(module.weight, pn, pk, partition, cores)
        packed = values[name + ".weight_packed"].to(torch.int64)
        q = ((packed.unsqueeze(-1) >> (torch.arange(8) * 4)) & 15).sub(8).reshape(n, k)
        assert torch.equal(decoded[:n, :k], q.to(torch.int8))
        assert torch.count_nonzero(decoded) == torch.count_nonzero(q)
