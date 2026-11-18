"""RTN/GPTQ checkpoint, UINT4 layout and cold admission checks without a board."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from rpu_backend.adapters.qwen3_5_moe.quant_scope import (
    GPTQ_METHOD, GPTQ_PROFILE_ID, RTN_METHOD, RTN_PROFILE_ID, RTN_RECIPE,
    validate_exact_profile,
)
from rpu_backend.adapters.qwen3_5_moe.rtn_checkpoint import RtnCheckpoint
from rpu_backend.adapters.qwen3_5_moe.gptq_checkpoint import GptqCheckpoint
from rpu_backend.quant.qwen3_5_moe_rtn import (
    pack_grouped_uint4, pack_rtn_qweight, quantize_rtn_group128,
    runtime_group_scales, unpack_rtn_matrix,
)
from test_grouped_uint4_metadata import inverse_metadata


def _inverse_grouped(packed, *, n, k, partition, cores=8):
    """Decode bytes and undo TP placement independently in NumPy."""
    e = packed.shape[0]
    nc, kc = (n // cores, k) if partition == "col" else (n, k // cores)
    ee, nn, kk = np.indices((e, n, k), dtype=np.int64)
    core = nn // nc if partition == "col" else kk // kc
    nl, kl = nn % nc, kk % kc
    offset = (((((ee * ((nc + 31) // 32) + nl // 32) * (kc // 32) + kl // 32)
                * cores + core) * 16 + nl % 16) * 2 + nl % 32 // 16) * 16 + kl % 16
    codes = (packed.numpy().reshape(-1)[offset] >> (kl % 32 // 16 * 4)) & 15
    return torch.from_numpy(codes.astype(np.uint8))


@pytest.mark.parametrize("partition,n,k", [("col", 512, 2048), ("row", 2048, 512)])
def test_real_expert_geometry_uint4_codes_roundtrip(partition, n, k):
    codes = torch.randint(0, 16, (2, n, k), generator=torch.Generator().manual_seed(15),
                          dtype=torch.uint8)
    packed = pack_grouped_uint4(codes, partition=partition)
    assert packed.dtype == torch.uint8 and packed.shape == (2, n * k // 2)
    assert torch.equal(_inverse_grouped(packed, n=n, k=k, partition=partition), codes)


def test_rtn_recipe_preserves_rounding_and_zero_scale_underflow():
    weight = torch.zeros(8, 128)
    weight[0, :6] = torch.tensor([-7.5, -6.5, -0.5, 0.5, 6.5, 7.5])
    weight[1, 0] = 1e-8
    tensors = quantize_rtn_group128(weight)
    codes, scales = unpack_rtn_matrix(**tensors, n=8, k=128)
    assert codes[0, :6].tolist() == [0, 2, 8, 8, 14, 15]
    assert scales[0, 0] == 1
    assert scales[0, 1] == 0 and codes[1, 0] == 15
    assert scales[0, 2] == torch.tensor(2 / 15).half()
    assert torch.all(codes[2:] == 8)
    # Code selection must use FP32 scale, before the stored FP16 cast.
    source = weight.numpy().reshape(8, 1, 128)
    a = np.max(np.abs(source), axis=-1, keepdims=True)
    s = np.where(a == 0, np.float32(2 / 15), a * np.float32(2 / 15))
    expected = np.clip(np.rint(source / s) + 8, 0, 15).astype(np.uint8).reshape(8, 128)
    assert np.array_equal(codes.numpy(), expected)


def test_gptq_container_bit_positions_and_down_group_refinement():
    codes = torch.arange(16, dtype=torch.uint8).repeat(8, 32)
    packed = pack_rtn_qweight(codes)
    assert packed[0, 0].item() == 0x76543210
    assert (packed[1, 0].item() & 0xFFFFFFFF) == 0xFEDCBA98
    scales = torch.tensor([[0., 1.], [2., 3.], [4., 5.], [6., 7.]], dtype=torch.float16)
    physical, group = runtime_group_scales(scales, k=512, partition="row")
    assert group == 64 and physical.shape == (8, 2)
    original = scales.repeat_interleave(128, dim=0)
    expanded = physical.repeat_interleave(64, dim=0)
    assert torch.equal(original.view(torch.int16), expanded.view(torch.int16))


@pytest.mark.parametrize("fault,match", [
    ("qzeros", "zero-minus-one"), ("g_idx", "act-order"),
    ("negative_scale", "nonnegative"), ("nan_scale", "finite"),
    ("qweight_shape", "qweight"),
])
def test_invalid_quantization_semantics_fail_before_packing(fault, match):
    tensors = quantize_rtn_group128(torch.ones(16, 128))
    if fault == "qzeros":
        tensors["qzeros"].fill_(-2004318072)  # official direct-8, not RTN zero-minus-one
    elif fault == "g_idx":
        tensors["g_idx"][0] = 1
    elif fault == "negative_scale":
        tensors["scales"][0, 0] = -1
    elif fault == "nan_scale":
        tensors["scales"][0, 0] = float("nan")
    else:
        tensors["qweight"] = tensors["qweight"][:, :-1].contiguous()
    with pytest.raises(ValueError, match=match):
        unpack_rtn_matrix(**tensors, n=16, k=128)


def _gptq_quant_config():
    return {
        "bits": 4, "group_size": 128, "sym": True, "desc_act": False,
        "quant_method": "gptq", "damp_percent": 0.01,
        "static_groups": False, "true_sequential": True,
        "dynamic": {"lm_head": {}, "model.language_model.embed_tokens": {},
                    "-:.*attn.*": {}, "-:.*shared_expert.*": {},
                    "-:.*mtp.*": {}, "-:.*visual.*": {}},
        "modules_to_not_convert": [],
    }


def _checkpoint(tmp_path, *, fault=None, profile="rtn"):

    """One tiny layer with each quartet split across independently named shards."""
    e, h, i = 2, 128, 512
    shards = [{}, {}]
    expected = {}
    gen = torch.Generator().manual_seed(23)
    for role, n, k in (("gate_proj", i, h), ("up_proj", i, h), ("down_proj", h, i)):
        expected[role] = []
        for expert in range(e):
            codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, generator=gen)
            scales = torch.rand(k // 128, n, generator=gen).half()
            scales[0, 0] = 0
            tensors = {"qweight": pack_rtn_qweight(codes), "scales": scales,
                       "qzeros": torch.full((k // 128, n // 8),
                                             0x77777777 if profile == "rtn" else -2004318072,
                                             dtype=torch.int32),
                       "g_idx": torch.arange(k, dtype=torch.int32) // 128}
            expected[role].append((codes, scales))
            for suffix, tensor in tensors.items():
                key = f"model.language_model.layers.0.mlp.experts.{expert}.{role}.{suffix}"
                shards[0 if suffix in ("qweight", "g_idx") else 1][key] = tensor
    shards[1]["model.language_model.norm.weight"] = torch.ones(h, dtype=torch.bfloat16)
    if fault == "missing":
        shards[1].pop("model.language_model.layers.0.mlp.experts.1.down_proj.qzeros")
    elif fault == "scope":
        shards[1]["model.language_model.layers.0.self_attn.q_proj.scales"] = torch.ones(1)
    elif fault == "shape":
        key = "model.language_model.layers.0.mlp.experts.0.gate_proj.scales"
        shards[1][key] = torch.ones(2, i, dtype=torch.float16)
    elif fault == "zero_encoding":
        key = "model.language_model.layers.0.mlp.experts.0.gate_proj.qzeros"
        shards[1][key].fill_(-2004318072 if profile == "rtn" else 0x77777777)
    elif fault == "alias":
        key = "model.language_model.layers.0.mlp.experts.0.gate_proj.qzeros"
        shards[1][key.replace("experts.0.", "experts.00.")] = shards[1][key].clone()
    elif fault == "mtp":
        # The real producer preserves 768 such BF16 MTP expert matrices.
        shards[1]["mtp.layers.0.mlp.experts.0.gate_proj.weight"] = torch.ones(
            i, h, dtype=torch.bfloat16)
    elif fault == "unexpected":
        shards[1]["unrecognized.weight"] = torch.ones(1, dtype=torch.bfloat16)
    weight_map, files, total = {}, {}, 0
    for number, tensors in enumerate(shards):
        filename = f"part-{7-number}.safetensors"
        path = tmp_path / filename
        save_file(tensors, path, metadata=({"format": "pt", "quantization_algorithm": "RTN"}
                                          if profile == "rtn" else None))
        files[filename] = {"bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        weight_map.update({key: filename for key in tensors})
        total += sum(value.nbytes for value in tensors.values())
    manifest = {"completed": True, "recipe": RTN_RECIPE, "quantized_matrices": 6,
                "quantizer_sha256": "a" * 64, "output_files": files}
    if profile == "rtn":
        (tmp_path / "derivation_manifest.json").write_text(json.dumps(manifest))
    else:
        (tmp_path / "config.json").write_text(json.dumps({
            "quantization_config": _gptq_quant_config()}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [11, 12]}))
    return expected


def _reader(path, profile="rtn"):
    cls = RtnCheckpoint if profile == "rtn" else GptqCheckpoint
    return cls(path, num_layers=1, num_experts=2, hidden_size=128,
                         intermediate_size=512)


@pytest.mark.parametrize("profile", ["rtn", "gptq"])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_sharded_quartets_preserve_zero_scales_and_per_expert_layout(tmp_path, profile, cores):
    expected = _checkpoint(tmp_path, profile=profile)
    source = _reader(tmp_path, profile).layer_source(0)
    assert source.profile_id == (RTN_PROFILE_ID if profile == "rtn" else GPTQ_PROFILE_ID)
    gate, up, down, gs, us, ds, mode = source.prepare(num_cores=cores)
    assert mode == 6
    for role, packed, scales, partition, n, k in (
        ("gate_proj", gate, gs, "col", 512, 128),
        ("up_proj", up, us, "col", 512, 128),
        ("down_proj", down, ds, "row", 128, 512),
    ):
        physical_n = 768 if cores == 6 and partition == "col" else n
        physical_k = 768 if cores == 6 and partition == "row" else k
        decoded = _inverse_grouped(packed, n=physical_n, k=physical_k, partition=partition, cores=cores)
        assert torch.equal(decoded[:, :n, :k], torch.stack([value[0] for value in expected[role]]))
        if cores == 6:
            tail = decoded[:, n:] if partition == "col" else decoded[:, :, k:]
            assert torch.all(tail == 8)
        reference = torch.stack([value[1] for value in expected[role]])
        if cores == 6:
            padded = torch.ones(reference.shape[0], physical_k // 128, physical_n, dtype=torch.float16)
            padded[:, :k // 128, :n] = reference
            reference = padded
        if partition == "row":
            reference = reference.repeat_interleave(2, dim=1)
        restored = inverse_metadata(scales, global_q=reference.shape[1],
                                    global_n=physical_n, partition=partition, cores=cores)
        assert torch.equal(restored.view(torch.int16), reference.view(torch.int16))
    source.release()
    with pytest.raises(ValueError, match="already consumed"):
        source.prepare()


@pytest.mark.parametrize("fault,match", [("missing", "incomplete"), ("scope", "scope"),
                                        ("shape", "expected"), ("alias", "scope")])
@pytest.mark.parametrize("profile", ["rtn", "gptq"])
def test_bad_index_or_tensor_metadata_rejected(tmp_path, fault, match, profile):
    _checkpoint(tmp_path, fault=fault, profile=profile)
    with pytest.raises(ValueError, match=match):
        _reader(tmp_path, profile)


@pytest.mark.parametrize("profile", ["rtn", "gptq"])
def test_lazy_value_validation_and_source_identity(tmp_path, profile):
    _checkpoint(tmp_path, fault="zero_encoding", profile=profile)
    reader = _reader(tmp_path, profile)
    with pytest.raises(ValueError, match="zero-minus-one|direct zero 8"):
        reader.layer_source(0).prepare()
    shard = next(iter(reader.shards.values()))
    with shard.path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="changed after admission"):
        reader.layer_source(0).prepare()


@pytest.mark.parametrize("fault", [None, "mtp", "unexpected"])
@pytest.mark.parametrize("profile", ["rtn", "gptq"])
def test_loader_materializes_regular_fp16_and_keeps_experts_lazy(tmp_path, monkeypatch, fault,
                                                               profile):
    from contextlib import contextmanager
    from rpu_backend.adapters.qwen3_5_moe import loader
    from rpu_backend.adapters.qwen3_5_moe import int4_checkpoint
    from rpu_backend.api.errors import RPUBackendError
    from rpu_backend.runtime.hw_attrs import validate_preinstall

    _checkpoint(tmp_path, fault=fault, profile=profile)
    original_safe_open = int4_checkpoint.safe_open
    requested = []

    @contextmanager
    def observed_open(*args, **kwargs):
        with original_safe_open(*args, **kwargs) as opened:
            def get_tensor(key):
                requested.append(key)
                return opened.get_tensor(key)
            yield SimpleNamespace(get_tensor=get_tensor)

    monkeypatch.setattr(int4_checkpoint, "safe_open", observed_open)

    class TinyModel(torch.nn.Module):
        # Same inherited declaration as the actual Transformers MoE model.
        _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            inner = self.model.language_model
            inner.norm = torch.nn.LayerNorm(128, bias=False)
            inner.layers = torch.nn.ModuleList([torch.nn.Module()])
            inner.layers[0].mlp = torch.nn.Module()
            inner.layers[0].mlp.experts = torch.nn.Module()
            experts = inner.layers[0].mlp.experts
            experts.gate_up_proj = torch.nn.Parameter(torch.empty(2, 1024, 128))
            experts.down_proj = torch.nn.Parameter(torch.empty(2, 128, 512))

        def tie_weights(self):
            pass

    config = SimpleNamespace(text_config=SimpleNamespace(
        num_hidden_layers=1, num_experts=2, hidden_size=128, moe_intermediate_size=512),
        quantization_config=_gptq_quant_config())
    seen = []

    def skeleton(config_copy):
        seen.append(config_copy)
        assert not hasattr(config_copy, "quantization_config")
        return TinyModel()

    monkeypatch.setattr(loader.AutoModelForImageTextToText, "from_config", skeleton)
    profile_id = RTN_PROFILE_ID if profile == "rtn" else GPTQ_PROFILE_ID
    method = RTN_METHOD if profile == "rtn" else GPTQ_METHOD
    monkeypatch.setattr(loader, "parse_quant_config", lambda _: {
        "method": method, "profile": profile_id})
    if fault == "unexpected":
        with pytest.raises(RPUBackendError, match="does not accept"):
            loader.load_qwen3_5_moe_model(tmp_path, config=config)
        assert not requested  # Reject unknown keys before any payload read.
        return
    model = loader.load_qwen3_5_moe_model(tmp_path, config=config)
    assert model.config is config and hasattr(config, "quantization_config")
    assert model.generation_config.eos_token_id == [11, 12]
    assert model.model.language_model.norm.weight.dtype == torch.float16
    experts = model.model.language_model.layers[0].mlp.experts
    assert experts.gate_up_proj.dtype == torch.uint8 and experts.gate_up_proj.numel() == 0
    assert experts.down_proj.device.type == "cpu" and experts.down_proj.numel() == 0
    assert experts._qwen3_5_moe_int4_source.consumed is False
    assert model._qwen3_5_moe_profile == profile_id
    assert experts._qwen3_5_moe_int4_source.profile_id == profile_id
    if profile == "gptq":
        descriptor = model._qwen3_5_moe_checkpoint_descriptor
        assert hashlib.sha256(json.dumps(descriptor, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest() == (
            model._qwen3_5_moe_checkpoint_identity)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert requested == ["model.language_model.norm.weight"]
    validate_preinstall(model)


def _rtn_config():
    from test_qwen35_moe_python_integration import _exact_config

    config = _exact_config()
    config.pop("quant_config")
    config["rtn_metadata"] = {"algorithm": "RTN", "manifest": "derivation_manifest.json",
                              "calibration": False}
    config["quantization_config"] = {"bits": 4, "group_size": 128, "sym": True,
                                     "desc_act": False, "quant_method": "gptq"}
    return config


def test_rtn_profile_admission_keeps_source_identity_at_each_core_count():
    from rpu_backend.adapters.qwen3_5_moe.cores import core_topology
    from rpu_backend.runtime import UnsupportedModelError

    config = _rtn_config()
    assert validate_exact_profile(config)["profile"] == RTN_PROFILE_ID
    assert core_topology(config=config).num_cores == 8
    for cores in (4, 6):
        assert core_topology({"model": {"num_cores": cores}}, config=config).num_cores == cores
    config.pop("rtn_metadata")
    with pytest.raises(ValueError):
        validate_exact_profile(config)


@pytest.mark.parametrize("field,value", [("sym", False), ("desc_act", True),
                                         ("bits", True), ("group_size", 64)])
def test_rtn_profile_rejects_other_quantization_semantics(field, value):
    config = _rtn_config()
    config["quantization_config"][field] = value
    with pytest.raises(ValueError, match="G128"):
        validate_exact_profile(config)


def test_rtn_profile_requires_boolean_calibration_metadata():
    config = _rtn_config()
    config["rtn_metadata"]["calibration"] = 0
    with pytest.raises(ValueError, match="derivation metadata"):
        validate_exact_profile(config)


def _accumulation_config(profile):
    from transformers import Qwen3_5MoeConfig
    from test_qwen35_moe_python_integration import _exact_config

    config = _exact_config() if profile == "fp8" else _rtn_config()
    if profile == "gptq":
        config.pop("rtn_metadata")
        config["quantization_config"] = _gptq_quant_config()
    return Qwen3_5MoeConfig.from_dict(config)


@pytest.mark.parametrize("profile", ["rtn", "gptq"])
@pytest.mark.parametrize("choice", [True, 0, 1, "true", None])
def test_int4_text_acc32_rejected_before_public_load_or_direct_install(monkeypatch, profile, choice):
    import rpu_backend
    from rpu_backend.api import conditional_generation as public
    from rpu_backend.api.errors import UnsupportedModelError
    from rpu_backend.adapters import qwen3_5_moe as moe

    config = _accumulation_config(profile)
    execution = {"prefill": {"linear_acc32": choice},
                 "vision": {"linear_acc32": True}}
    monkeypatch.setattr(public.AutoConfig, "from_pretrained", lambda *a, **k: config)
    monkeypatch.setattr(moe.Qwen3_5MoeAdapter, "load_pretrained",
                        lambda *a, **k: pytest.fail("unsupported accumulation read weights"))
    monkeypatch.setattr(rpu_backend.graph, "GraphCache",
                        lambda *a, **k: pytest.fail("unsupported accumulation created GraphCache"))
    error, message = ((UnsupportedModelError, "Text ACC16 only") if choice is True
                      else (TypeError, "linear_acc32.*must be bool"))
    with pytest.raises(error, match=message):
        public.RPUModelForConditionalGeneration.from_pretrained(
            f"/mock-{profile}", rpu_execution=execution)
    inner = SimpleNamespace()
    with pytest.raises(error, match=message):
        moe.install_qwen3_5_moe_text_for_rpu(
            inner, config, max_seq_len=64, execution_config=execution)
    assert not hasattr(inner, "_rpu_qwen3_5_moe_text_install_started")
    assert not hasattr(inner, "_rpu_qwen3_5_moe")


@pytest.mark.parametrize("profile,text", [
    (profile, text) for profile in ("fp8", "rtn", "gptq")
    for text in ({}, {"linear_acc32": False})
] + [("fp8", {"linear_acc32": True})])
def test_moe_text_accumulation_keeps_fp8_and_independent_vision_modes(profile, text):
    from rpu_backend.adapters.qwen3_5_moe import Qwen3_5MoeAdapter
    from rpu_backend.adapters.qwen3_5_moe.text import (
        _validate_qwen3_5_moe_linear_accumulation,
    )

    execution = {"prefill": text, "vision": {"linear_acc32": True}}
    config = _accumulation_config(profile)
    Qwen3_5MoeAdapter.preflight_execution(config, execution)
    selected_profile = validate_exact_profile(config)["profile"]
    assert _validate_qwen3_5_moe_linear_accumulation(
        selected_profile, execution) is text.get("linear_acc32", False)
    assert execution["vision"]["linear_acc32"] is True


@pytest.mark.parametrize("profile", ["rtn", "gptq"])
def test_layer_packing_releases_source_only_after_success(tmp_path, monkeypatch, profile):
    from rpu_backend.adapters.qwen3_5_moe import weights

    _checkpoint(tmp_path, profile=profile)
    source = _reader(tmp_path, profile).layer_source(0)
    linear = lambda n, k: SimpleNamespace(weight=torch.ones(n, k, dtype=torch.float16))
    mlp = SimpleNamespace(
        experts=SimpleNamespace(_qwen3_5_moe_int4_source=source,
                                rpu_weight_mode=6, rpu_quant_profile=source.profile_id),
        gate=linear(256, 128),
        shared_expert=SimpleNamespace(gate_proj=linear(512, 128),
                                     up_proj=linear(512, 128), down_proj=linear(128, 512)),
        shared_expert_gate=linear(1, 128),
    )
    # Exercise the real ordinary FP16 transform too; its device copy is CPU.
    mlp.shared_expert.gate_proj.weight = mlp.shared_expert.gate_proj.weight.float()
    with pytest.raises(TypeError, match="remain FP16"):
        weights.prepare_moe_layer_weights(mlp, device="cpu")
    assert source.consumed is False
    mlp.shared_expert.gate_proj.weight = mlp.shared_expert.gate_proj.weight.half()
    packed = weights.prepare_moe_layer_weights(mlp, device="cpu")
    assert packed.routed_weight_mode == 6 and source.consumed is True
    assert packed.routed_gate.shape == (2, 512 * 128 // 2)
    assert packed.routed_gate_scale.shape == (2, 1, 1, 8, 4, 64)
    assert packed.routed_down_scale.shape == (2, 1, 2, 8, 4, 64)
    for name in ("router", "shared_gate", "shared_up", "shared_down", "shared_scalar_gate"):
        assert getattr(packed, name).dtype == torch.float16


@pytest.mark.parametrize("profile,cores,acc32,message", [
    ("fp8", 8, True, "resident prefill requires Text ACC16"),
    *[(profile, cores, False, "resident prefill requires model.num_cores=8")
      for profile in ("fp8", "rtn", "gptq") for cores in (4, 6)],
])
def test_resident_invalid_cold_mode_rejects_before_public_load_or_install(
        monkeypatch, profile, cores, acc32, message):
    import rpu_backend
    from rpu_backend.api import conditional_generation as public
    from rpu_backend.api.errors import UnsupportedModelError
    from rpu_backend.adapters import qwen3_5_moe as moe
    from rpu_backend.runtime import weights

    config = _accumulation_config(profile)
    execution = {"model": {"num_cores": cores},
                 "prefill": {"moe_resident_prefill": True, "linear_acc32": acc32}}
    monkeypatch.setattr(public.AutoConfig, "from_pretrained", lambda *a, **k: config)
    for owner, name in ((moe.Qwen3_5MoeAdapter, "load_pretrained"),
                        (rpu_backend.graph, "GraphCache"),
                        (rpu_backend.graph, "Graph"),
                        (weights, "convert_linear_weights_inplace")):
        monkeypatch.setattr(owner, name,
            lambda *a, **k: pytest.fail("invalid resident request reached loading/arena/swizzle"))
    with pytest.raises(UnsupportedModelError, match=message):
        moe.Qwen3_5MoeAdapter.preflight_execution(config, execution)
    with pytest.raises(UnsupportedModelError, match=message):
        public.RPUModelForConditionalGeneration.from_pretrained(
            f"/mock-{profile}", rpu_execution=execution)
    inner = SimpleNamespace()
    with pytest.raises(UnsupportedModelError, match=message):
        moe.install_qwen3_5_moe_text_for_rpu(
            inner, config, max_seq_len=64, execution_config=execution)
    assert vars(inner) == {}


@pytest.mark.parametrize("profile,cores,resident,acc32", [
    *[(profile, 8, True, False) for profile in ("fp8", "rtn", "gptq")],
    *[("fp8", cores, False, acc32) for cores in (4, 6, 8) for acc32 in (False, True)],
    *[(profile, cores, False, False) for profile in ("rtn", "gptq") for cores in (4, 6, 8)],
])
def test_resident_cold_admission_keeps_legal_tp8_and_ordinary_modes(
        monkeypatch, profile, cores, resident, acc32):
    from rpu_backend.adapters import qwen3_5_moe as moe
    from rpu_backend.adapters.qwen3_5_moe import text

    config = _accumulation_config(profile)
    execution = {"model": {"num_cores": cores},
        "prefill": {"moe_resident_prefill": resident, "linear_acc32": acc32},
        "vision": {"linear_acc32": True}}
    moe.Qwen3_5MoeAdapter.preflight_execution(config, execution)
    assert text._validate_qwen3_5_moe_linear_accumulation(
        validate_exact_profile(config)["profile"], execution) is acc32
    # The actual direct installer must pass admission and reach option resolution.
    # Stop there: no graph allocation, transform, upload or native invocation.
    class ReachedOptions(Exception):
        pass
    def accepted(*args, **kwargs):
        raise ReachedOptions()
    monkeypatch.setattr(text, "_resolve_text_install_options", accepted)
    for name in ("qwen3_5_moe_set_execution_cores", "qwen3_5_moe_get_execution_topology",
                 "qwen3_5_moe_get_execution_topology_v2"):
        monkeypatch.setattr(torch.ops.rpu, name,
            lambda *a, **k: pytest.fail("cold admission invoked native"), raising=False)
    inner = SimpleNamespace()
    with pytest.raises(ReachedOptions):
        moe.install_qwen3_5_moe_text_for_rpu(
            inner, config, max_seq_len=64, execution_config=execution)
    assert vars(inner) == {} and execution["vision"]["linear_acc32"] is True


@pytest.mark.parametrize("partition,cores,n,k", [
    ("col", 8, 128, 128), ("col", 4, 192, 96), ("row", 6, 48, 576),
])
def test_grouped_w4_physical_tail_roundtrip_preserves_expert_offsets(partition, cores, n, k):
    from rpu_backend.quant.int4_pgrp_pack import w4_storage_shape

    codes = torch.randint(0, 16, (3, n, k), generator=torch.Generator().manual_seed(1021), dtype=torch.uint8)
    packed = pack_grouped_uint4(codes, partition=partition, num_cores=cores)
    physical_n, physical_k = w4_storage_shape(n, k, int(partition == "col"), cores)
    assert packed.shape == (3, physical_n * physical_k)
    assert torch.equal(_inverse_grouped(packed, n=n, k=k, partition=partition, cores=cores), codes)
