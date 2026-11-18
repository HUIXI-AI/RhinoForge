"""Reduced quantized layouts preserve logical weights and group boundaries."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3_5Config

from rpu_backend.adapters.qwen3_5.cores import (
    text_mlp_intermediate_size, validate_core_profile,
)
from rpu_backend.adapters.qwen3_5.weights import (
    pad_quant_mlp_projection, pad_quant_mlp_scale,
)
from rpu_backend.quant.convert_qwen3_5 import _quant_config
from rpu_backend.quant.int4_pgrp_pack import (
    dequantize_int4_group_wise, swizzle_int4_pgrp_scale, swizzle_pack_int4_pgrp,
)
from test_qwen35_core_profiles import _config
from test_qwen35_dense27_profiles import config27


@pytest.mark.parametrize("heads,hidden", [(16, 1024), (16, 2048), (32, 2560),
                                         (32, 4096), (48, 5120)])
@pytest.mark.parametrize("root_cores", [4, 6, 8])
def test_actual_gdn_scalar_preparation_keeps_logical_n16_and_physical_n32(
        heads, hidden, root_cores, monkeypatch):
    # Execute the production nested preparation functions; replace only the
    # device upload. This catches the actual Python/native install boundary.
    source = Path(__file__).resolve().parents[1] / "python/rpu_backend/adapters/qwen3_5/text.py"
    tree = ast.parse(source.read_text())
    install = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "_install_qwen3_5_text_for_rpu_impl")
    names = {"_prep_weight", "_ba_nbg", "_ba_nbg_scale"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "_scale_to_rpu"]
    nodes += [node for node in install.body
              if (isinstance(node, ast.FunctionDef) and node.name in names)
              or (isinstance(node, ast.Assign) and any(
                  isinstance(target, ast.Name) and target.id in {"_vg_c", "_n_bg"}
                  for target in node.targets))]
    cores = 8 if root_cores == 8 else 4
    scope = dict(torch=torch, NUM_CORES=cores, _gnvh=heads, cpu_stage_weights=True,
                 NVFP4="nvfp4", W4A16_AWQ="w4a16_awq", W4A16_PGRP="w4a16_pgrp")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    original_to = torch.Tensor.to

    def cpu_upload(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            args = ("cpu", *args[1:])
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_upload)
    spec = SimpleNamespace(method="w4a16_pgrp", group_size=32)
    codes = ((torch.arange(heads * hidden).reshape(heads, hidden) % 13) - 6).to(torch.int8)
    scales = (torch.arange(heads * (hidden // 32)).reshape(heads, hidden // 32)
              .remainder(127).add(1).half() / 128)
    weight = scope["_ba_nbg"](codes, spec)
    scale = scope["_ba_nbg_scale"](scales, spec)
    assert scope["_n_bg"] == 16
    assert weight.dtype == torch.uint8 and weight.shape == (32 * cores, hidden // 2)
    assert weight.is_contiguous() and scale.is_contiguous()

    # Independently decode every physical N32 row, including its zero tail.
    cc, nn, kk = torch.meshgrid(torch.arange(cores), torch.arange(32),
                               torch.arange(hidden), indexing="ij")
    addr = (((((nn // 32) * (hidden // 32) + kk // 32) * cores + cc)
             * 16 + nn % 16) * 2 + nn % 32 // 16) * 16 + kk % 16
    decoded = ((weight.reshape(-1)[addr] >> ((kk % 32 // 16) * 4)) & 15).to(torch.int8)
    decoded[decoded >= 8] -= 16
    local_heads = heads // cores
    assert torch.equal(decoded[:, :local_heads], codes.reshape(cores, local_heads, hidden))
    assert torch.count_nonzero(decoded[:, local_heads:]) == 0

    groups = hidden // 32
    group_blocks = (groups + 3) // 4
    assert scale.shape == (32, group_blocks * cores * 4 * 64 // 32)
    decoded_scale = scale.reshape(group_blocks, 1, cores, 4, 64).permute(
        2, 0, 3, 1, 4).reshape(cores, group_blocks * 4, 64)
    assert torch.equal(decoded_scale[:, :groups, :local_heads],
                       scales.reshape(cores, local_heads, groups).transpose(1, 2))
    assert torch.count_nonzero(decoded_scale[:, :, local_heads:]) == 0
    assert torch.count_nonzero(decoded_scale[:, groups:, :]) == 0


@pytest.mark.parametrize("size", [0.8, 2, 4, 9, 27])
@pytest.mark.parametrize("method", ["w8a16", "w4a16_pgrp"])
@pytest.mark.parametrize("cores", [4, 6])
def test_dense_quantized_profiles_use_aligned_component_widths(size, method, cores):
    config = config27(method) if size == 27 else _config(size)
    config.quant_config = {"method": method}
    if method == "w4a16_pgrp":
        config.quant_config["group_size"] = 32
    topology = validate_core_profile(config, {"model": {"num_cores": cores}})
    physical = text_mlp_intermediate_size(config, topology)
    assert topology.attn_tp == 4 and topology.lm_head_tp == cores
    assert topology.mlp_tp == cores
    stripe = 32 if method == "w4a16_pgrp" else 32
    assert physical % ((32 if method == "w4a16_pgrp" else stripe) * cores) == 0
    assert physical >= config.text_config.intermediate_size
    if method == "w4a16_pgrp" and cores == 6:
        assert physical == {0.8: 3648, 2: 6144, 4: 9216, 9: 12288, 27: 17472}[size]


@pytest.mark.parametrize("recipe", ["w8_main_w8_gdn", "w4_main_w8_gdn",
                                     "w4_mlp_only", "w4_mlp_w8_attn_gdn"])
@pytest.mark.parametrize("cores", [4, 6])
def test_named_mixed_recipes_keep_their_precision_on_reduced_components(recipe, cores):
    config = _config(2)
    full = [kind == "full_attention" for kind in config.text_config.layer_types]
    config.quant_config = _quant_config(full, None, 32, profile=recipe)
    assert validate_core_profile(config, {"model": {"num_cores": cores}}).mlp_tp == cores


@pytest.mark.parametrize("size", [2, 4, 9])
@pytest.mark.parametrize("cores", [4, 6])
def test_original_awq_metadata_selects_reduced_components(size, cores):
    fixture = Path(__file__).parent / "fixtures/qwen35_quanttrio_awq" / f"{size}b.json"
    config = Qwen3_5Config.from_dict(json.loads(fixture.read_text()))
    topology = validate_core_profile(config, {"model": {"num_cores": cores}})
    assert (topology.attn_tp, topology.mlp_tp) == (4, cores)
    assert config.text_config.intermediate_size % (128 * cores) == 0


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_original_awq_metadata_admits_official_27b_geometry(cores):
    from rpu_backend.adapters.qwen3_5.quant_scope import (
        quanttrio_awq_config, resolve_quant_specs, validate_modern_27b_profile,
    )

    config = config27()
    config.quant_config = None
    config.quantization_config = {
        "quant_method": "awq", "bits": 4, "group_size": 128,
        "version": "gemm", "zero_point": True,
        "modules_to_not_convert": [
            "visual", "linear_attn", "self_attn", "model.layers.0.", "mtp",
        ],
    }
    policy = quanttrio_awq_config(config)
    assert policy is not None
    assert not validate_modern_27b_profile(config, allow_fp16=True)
    is_full = [kind == "full_attention" for kind in config.text_config.layer_types]
    specs = resolve_quant_specs(is_full, policy)
    assert len(specs) == 496
    assert sum(spec.method == "w4a16_awq" for spec in specs.values()) == 63 * 3
    topology = validate_core_profile(config, {"model": {"num_cores": cores}})
    assert topology.num_cores == cores
    assert text_mlp_intermediate_size(config, topology) == (
        17664 if cores == 6 else 17408)


@pytest.mark.parametrize("logical,physical", [(3584, 3648), (17408, 17472)])
@pytest.mark.parametrize("role", ["gate_proj", "down_proj"])
def test_w4_padding_preserves_dequantized_values_and_packs_six_core_stripes(logical, physical, role):
    # A compact independent hidden dimension keeps real packing affordable.
    n, k = (64, logical) if role == "down_proj" else (logical, 64)
    codes = ((torch.arange(n * k).reshape(n, k) % 16) - 8).to(torch.int8)
    scale = (torch.arange((k // 32) * n).reshape(k // 32, n) % 11 + 1).half() / 32
    expected = dequantize_int4_group_wise(codes, scale)
    padded = pad_quant_mlp_projection(codes, projection_name=role,
        logical_size=logical, physical_size=physical)
    padded_scale = pad_quant_mlp_scale(scale.t().contiguous(), projection_name=role,
        logical_size=logical, physical_size=physical, group_size=32).t().contiguous()
    actual = dequantize_int4_group_wise(padded, padded_scale)
    torch.testing.assert_close(actual[:n, :k], expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[n:, :]) == 0
    assert torch.count_nonzero(actual[:, k:]) == 0
    partition = 0 if role == "down_proj" else 1
    packed = swizzle_pack_int4_pgrp(padded, partition, 6)
    packed_scale = swizzle_int4_pgrp_scale(padded_scale, 32, partition, 6)
    assert packed.dtype == torch.uint8 and packed.numel() == padded.numel() // 2
    assert packed_scale.is_contiguous() and packed_scale.shape[0] == 32


def test_awq_27b_padding_accepts_uint4_group128_payloads():
    from rpu_backend.adapters.qwen3_5.weights import (
        pad_quant_mlp_projection, pad_quant_mlp_scale,
    )

    logical, physical = 17408, 17664
    weight = torch.zeros((logical, 128), dtype=torch.uint8)
    scale = torch.ones((logical, 40), dtype=torch.float16)
    padded_weight = pad_quant_mlp_projection(
        weight, projection_name="gate_proj",
        logical_size=logical, physical_size=physical)
    padded_scale = pad_quant_mlp_scale(
        scale, projection_name="gate_proj",
        logical_size=logical, physical_size=physical, group_size=128)
    assert padded_weight.shape == (physical, 128)
    assert padded_scale.shape == (physical, 40)
    assert torch.count_nonzero(padded_weight[logical:]) == 0
    assert torch.all(padded_scale[logical:] == 1)


@pytest.mark.parametrize("partition", [0, 1])
@pytest.mark.parametrize("cores", [4, 6])
def test_awq_reduced_packing_keeps_unsigned_codes_and_direct_zero(partition, cores):
    from rpu_backend.quant.awq_gemm import swizzle_pack_uint4_pgrp, swizzle_awq_negative_zero
    n, k = 192, 768
    # Vary both axes so N16/N32 and K32/K64 permutations cannot alias.
    codes = ((torch.arange(n)[:, None] * 131 + torch.arange(k)[None, :] * 7)
             // 13).remainder(16).to(torch.uint8)
    zeros = (torch.arange((k // 128) * n).reshape(k // 128, n) % 16).to(torch.uint8)
    if partition == 0 and cores == 4:
        k = 1024
        codes = codes[:, :1].expand(n, k).contiguous()
        zeros = zeros[:1].expand(k // 128, n).contiguous()
    packed = swizzle_pack_uint4_pgrp(codes, partition, cores)
    # Independent scalar address oracle for published BASE N32K32. The
    # col-TP4 case has localN48 and must preserve its zero-padded N32 tail.
    local_n, local_k = (n // cores, k) if partition else (n, k // cores)
    expected = []
    for nb in range((local_n + 31) // 32):
        for kb in range(local_k // 32):
            for core in range(cores):
                for thread in range(16):
                    for half_n in (0, 16):
                        nl = nb * 32 + half_n + thread
                        ng = nl + (core * local_n if partition else 0)
                        for lane in range(16):
                            kg = kb * 32 + lane + (0 if partition else core * local_k)
                            expected.append(0 if nl >= local_n else
                                int(codes[ng, kg]) | (int(codes[ng, kg + 16]) << 4))
    assert torch.equal(packed.flatten(), torch.tensor(expected, dtype=torch.uint8))
    assert torch.equal(swizzle_awq_negative_zero(zeros, partition, cores),
                       swizzle_int4_pgrp_scale(-zeros.half(), 128, partition, cores))
