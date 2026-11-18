"""Qwen3.5-35B-A3B routed-only W8A16 (grouped mode 1); no weights or RPU required."""
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tomllib
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3_5MoeConfig

from _support_status import expected_support_status
from rpu_backend.adapters.qwen3_5_moe import loader, quant_scope
from rpu_backend.quant import convert_qwen3_5_moe_fp8 as fp8_converter
from rpu_backend.quant import convert_qwen3_5_moe_int8 as converter
from rpu_backend.quant import precision_recipes
from test_convert_qwen35_moe_fp8 import Skeleton, source

ROOT = Path(__file__).resolve().parents[1]
TEXT_CONFIG = ROOT / "examples/configs/qwen3_5/text/35b_a3b/w8a16_experts.toml"
VL_CONFIG = ROOT / "examples/configs/qwen3_5/vl/35b_a3b/w8a16_experts.toml"
NATIVE = ROOT / "src/fused/rpu_qwen3_5_moe_model.cpp"


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    return importlib.import_module("examples.common.config")


def _exact_int8_metadata(common):
    """The exact 35B-A3B config.json the converter writes, without weights."""
    scope = common._backend_leaf("adapters/qwen3_5_moe/quant_scope.py")
    text = dict(scope._EXACT_TEXT_FIELDS)
    text["layer_types"] = (["linear_attention"] * 3 + ["full_attention"]) * 10
    text["rope_parameters"] = dict(scope._EXACT_ROPE_FIELDS)
    names = sorted(scope.expected_int8_weight_names(text["layer_types"]))
    return {
        "model_type": "qwen3_5_moe", "tie_word_embeddings": False,
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": text, "vision_config": dict(scope._EXACT_VISION_FIELDS),
        "quant_config": {
            "method": scope.INT8_METHOD, "format": scope.INT8_FORMAT, "weight_mode": 1,
            "qaxis": -1, "scale_dtype": "float16", "scope": scope.INT8_SCOPE,
            "profile": scope.INT8_PROFILE_ID, "symmetric": True, "calibration": False,
            "fp16_preserve": list(scope.INT8_FP16_PRESERVE),
            "quantized_tensor_count": len(names), "quantized_tensors": names,
        },
    }


# ---------------------------------------------------------------- public profile
@pytest.mark.parametrize("path,component,target", [
    (TEXT_CONFIG, "text", "qwen3_5"), (VL_CONFIG, "multimodal", "qwen3_5_vision")])
def test_int8_public_profiles_bind_exact_recipe_and_cores(common, monkeypatch, tmp_path,
                                                          path, component, target):
    config = common.load_config(path)
    profile = common.planner_profiles()[config["example"]["profile_id"]]
    limits = profile["input_envelope"]["limits"]
    assert profile["id"] == f"qwen3_5-35b-a3b.rtn-w8a16.{component}"
    assert profile["support"]["status"] == expected_support_status(profile, "IMPLEMENTED_UNVERIFIED_HW")
    assert profile["hw_status"] == "IMPLEMENTED_UNVERIFIED_HW"
    assert (profile["precision"], profile["component"]) == ("w8a16", component)
    assert limits["checkpoint_profile"] == quant_scope.INT8_PROFILE_ID
    assert limits["allowed_num_cores"] == [4, 6, 8] and "group_size" not in limits
    assert limits["formal_certification"] is False and limits["batch_size"] == 1
    assert config["rpu_execution"]["prefill"]["linear_acc32"] is False
    assert config["rpu_execution"]["prefill"].get("moe_resident_prefill", False) is False
    recipe = precision_recipes.for_profile("qwen3_5", profile["id"])
    assert recipe.name == "w8a16_experts" == tomllib.loads(path.read_text())["model"]["precision"]
    assert config["example"]["dtype"] == recipe.label == "w8a16"
    assert recipe.spec("experts").format == "w8a16"
    assert {recipe.spec(name).format for name in ("attention", "gdn", "mlp")} == {"fp16"}
    assert precision_recipes.checkpoint_recipe(recipe) == (converter.FAMILY, converter.RECIPE)
    assert quant_scope.runtime_capabilities(quant_scope.INT8_PROFILE_ID).allowed_cores == (4, 6, 8)
    for cores in (4, 6, 8):
        execution = deepcopy(config["rpu_execution"])
        execution["model"]["num_cores"] = cores
        common._validate_execution(execution, profile=profile)
    registry = importlib.util.spec_from_file_location(
        "int8_model_registry", ROOT / "python/rpu_backend/model_registry.py")
    module = importlib.util.module_from_spec(registry)
    monkeypatch.setitem(sys.modules, "int8_model_registry", module)
    registry.loader.exec_module(module)
    monkeypatch.setenv("RPU_MODEL_CACHE", str(tmp_path))
    assert module.model_path(config["example"]["registry_alias"]) == tmp_path / "Qwen3.5-35B-A3B-RTN-W8A16"


@pytest.mark.parametrize("field,value,match", [
    ("cores", 5, "model.num_cores"), ("cores", 7, "model.num_cores"),
    ("acc32", True, "ACC16 only"), ("resident", True, "resident prefill"),
    ("alias", "qwen3_5-35b-a3b-rtn-w4a16", "INT8"), ("component", "multimodal", "INT8"),
    ("group_size", 128, "INT8"),
])
def test_int8_config_rejects_wrong_core_accumulation_or_profile(common, field, value, match):
    config = common.load_config(TEXT_CONFIG)
    profile = deepcopy(common.planner_profiles()[config["example"]["profile_id"]])
    execution = config["rpu_execution"]
    if field == "cores":
        execution["model"]["num_cores"] = value
    elif field == "acc32":
        execution["prefill"]["linear_acc32"] = value
    elif field == "resident":
        execution["prefill"]["moe_resident_prefill"] = value
    elif field == "group_size":
        profile["input_envelope"]["limits"]["group_size"] = value
    else:
        profile[{"alias": "asset_alias", "component": "component"}[field]] = value
    with pytest.raises(ValueError, match=match):
        common._validate_execution(execution, profile=profile)


# ---------------------------------------------------------- checkpoint identity
def test_int8_checkpoint_binds_only_its_own_profile(common, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_exact_int8_metadata(common)))
    int8 = common.load_config(TEXT_CONFIG)
    common.validate_checkpoint_profile(int8, str(tmp_path))
    for other in ("w4a16_experts_rtn_g128.toml", "fp8_e4m3_experts_attn_gdn.toml"):
        config = common.load_config(TEXT_CONFIG.with_name(other))
        # The mixed-E4M3 profile binds its default checkpoint profile on --model PATH.
        config["_model_override"] = str(tmp_path)
        with pytest.raises(common.ConfigError, match="rejects checkpoint|requires checkpoint profile"):
            common.validate_checkpoint_profile(config, str(tmp_path))


@pytest.mark.parametrize("mutation", [
    "bool_mode", "fp8_mode", "unknown_field", "missing_tensor", "fp16_preserve",
    "asymmetric", "count"])
def test_int8_declaration_mutations_fail_closed(common, mutation):
    data = _exact_int8_metadata(common)
    quant = data["quant_config"]
    if mutation == "bool_mode":
        quant["weight_mode"] = True
    elif mutation == "fp8_mode":
        quant["weight_mode"] = 5
    elif mutation == "unknown_field":
        quant["group_size"] = 128
    elif mutation == "missing_tensor":
        quant["quantized_tensors"] = quant["quantized_tensors"][1:]
        quant["quantized_tensor_count"] -= 1
    elif mutation == "fp16_preserve":
        quant["fp16_preserve"] = list(quant_scope.FP16_PRESERVE)
    elif mutation == "asymmetric":
        quant["symmetric"] = False
    else:
        quant["quantized_tensor_count"] = 81
    with pytest.raises(ValueError):
        quant_scope.validate_exact_profile(data)


# ------------------------------------------------------- converter and loader
def _convert(tmp_path, monkeypatch, *, sharded=True):
    src, original = source(tmp_path, sharded=sharded)
    # Shrink geometry only; keep the real scope, converter and loader machinery.
    monkeypatch.setattr(fp8_converter, "validate_exact_profile", lambda *a, **kw: None)
    dst = tmp_path / "output"
    return src, original, dst, converter.convert_checkpoint(src, dst)


@pytest.mark.parametrize("sharded", [True, False])
def test_convert_and_load_keep_routed_int8_codes_and_fp16_mixers(tmp_path, monkeypatch, sharded):
    src, original, dst, stats = _convert(tmp_path, monkeypatch, sharded=sharded)
    cfg = Qwen3_5MoeConfig.from_dict(json.loads((dst / "config.json").read_text()))
    layer_types = cfg.text_config.layer_types
    quant = quant_scope.validate_int8_config(cfg.quant_config, layer_types)
    routed = set(quant_scope.expected_int8_weight_names(layer_types))
    assert stats["n_quantized"] == len(routed) == 4
    assert precision_recipes.read_record(dst) == ("qwen3_5", "w8a16_experts")
    monkeypatch.setattr(loader, "parse_quant_config", lambda _: quant)
    monkeypatch.setattr(loader.AutoModelForImageTextToText, "from_config",
                        lambda _: Skeleton(original))
    model = loader.load_qwen3_5_moe_model(dst, config=cfg)
    assert model._qwen3_5_moe_profile == quant_scope.INT8_PROFILE_ID
    parameters = dict(model.named_parameters())
    for name, expected in original.items():
        actual = parameters[name]
        if name not in routed:
            # Attention/GDN mixers stay FP16 here, unlike the mixed-E4M3 bundle.
            assert actual.dtype == torch.float16 and torch.equal(actual, expected.half())
            continue
        module_name, parameter_name = name.rsplit(".", 1)
        module = model.get_submodule(module_name)
        scale = getattr(module, parameter_name + "_scale")
        assert actual.dtype == torch.int8 and scale.dtype == torch.float16
        assert (module.rpu_weight_mode, module.rpu_int8_scale_axis,
                module.rpu_quant_profile) == (1, -1, quant_scope.INT8_PROFILE_ID)
        assert not hasattr(module, "rpu_fp8_format")
        # Independent symmetric RTN oracle: max|w|/127 per output row.
        reference = expected.float()
        expected_scale = (reference.abs().amax(-1) / 127).half()
        assert torch.equal(scale, expected_scale)
        codes = torch.round(reference / expected_scale.float().unsqueeze(-1)).clamp(-127, 127)
        assert torch.equal(actual.float(), codes)
        assert int(actual.abs().amax()) == 127
    assert model.generation_config.eos_token_id == [11, 12]
    for filename in ("generation_config.json", "tokenizer_config.json", "preprocessor_config.json"):
        assert (src / filename).read_bytes() == (dst / filename).read_bytes()


def test_loader_rejects_e4m3_bytes_relabelled_as_int8(tmp_path, monkeypatch):
    _, original, dst, _ = _convert(tmp_path, monkeypatch, sharded=False)
    cfg = Qwen3_5MoeConfig.from_dict(json.loads((dst / "config.json").read_text()))
    quant = quant_scope.validate_int8_config(cfg.quant_config, cfg.text_config.layer_types)
    tensors = load_file(dst / "model.safetensors")
    name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    tensors[name] = tensors[name].view(torch.uint8)
    save_file(tensors, dst / "model.safetensors")
    monkeypatch.setattr(loader, "parse_quant_config", lambda _: quant)
    monkeypatch.setattr(loader.AutoModelForImageTextToText, "from_config",
                        lambda _: Skeleton(original))
    with pytest.raises(Exception, match="must be torch.int8"):
        loader.load_qwen3_5_moe_model(dst, config=cfg)


@pytest.mark.parametrize("fault,match", [
    ("missing", "complete routed-expert inventory"), ("nan", "nonfinite"),
    ("quant_key", "unexpected quantization key"), ("int_dtype", "FP16/BF16/FP32"),
    ("shape", "shape .* must be"),
])
def test_bad_sources_fail_without_publishing_output(tmp_path, monkeypatch, fault, match):
    src, _ = source(tmp_path, sharded=False)
    monkeypatch.setattr(fp8_converter, "validate_exact_profile", lambda *a, **kw: None)
    name = "model.language_model.layers.1.mlp.experts.down_proj"
    tensors = load_file(src / "model.safetensors")
    if fault == "missing":
        tensors.pop(name)
    elif fault == "nan":
        tensors[name][1, 0, 0] = float("nan")
    elif fault == "quant_key":
        tensors["model.language_model.norm.weight_scale"] = torch.ones(1)
    elif fault == "int_dtype":
        tensors[name] = tensors[name].to(torch.int8)
    else:
        tensors[name] = tensors[name][:, :, :-1].contiguous()
    save_file(tensors, src / "model.safetensors")
    with pytest.raises((ValueError, TypeError), match=match):
        converter.convert_checkpoint(src, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


# ------------------------------------------------------------ cold preparation
def _int8_experts(e, h, logical, generator):
    return SimpleNamespace(
        gate_up_proj=torch.randint(-127, 128, (e, logical * 2, h), dtype=torch.int8,
                                   generator=generator),
        down_proj=torch.randint(-127, 128, (e, h, logical), dtype=torch.int8, generator=generator),
        gate_up_proj_scale=torch.rand(e, logical * 2, generator=generator).half() + 0.25,
        down_proj_scale=torch.rand(e, h, generator=generator).half() + 0.25,
        rpu_weight_mode=1, rpu_int8_scale_axis=-1, rpu_quant_profile=quant_scope.INT8_PROFILE_ID)


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_int8_routed_packing_keeps_signed_codes_and_tp6_zero_lanes(cores):
    from rpu_backend.adapters.qwen3_5_moe.weights import _prepare_routed

    generator = torch.Generator().manual_seed(808)
    e, h, logical = 2, 64, 512
    physical = 576 if cores == 6 else logical
    module = _int8_experts(e, h, logical, generator)
    source_tensors = [t.clone() for t in (module.gate_up_proj, module.down_proj,
                                          module.gate_up_proj_scale, module.down_proj_scale)]
    gate, up, down, gs, us, ds, mode = _prepare_routed(module, cores)
    assert mode == 1 and gate.dtype == up.dtype == down.dtype == torch.int8

    # Decode the grouped byte ABI independently of the packer.
    def unpack_col(tensor):
        return tensor.view(e, physical // cores // 16, h // 32, cores, 16, 32).permute(
            0, 3, 1, 4, 2, 5).contiguous().view(e, physical, h)
    raw_down = down.view(e, h // 16, physical // cores // 32, cores, 16, 32).permute(
        0, 1, 4, 3, 2, 5).contiguous().view(e, h, physical)
    for packed, half in ((unpack_col(gate), slice(None, logical)),
                         (unpack_col(up), slice(logical, None))):
        assert torch.equal(packed[:, :logical], source_tensors[0][:, half])
        assert torch.count_nonzero(packed[:, logical:]) == 0
    assert torch.equal(raw_down[:, :, :logical], source_tensors[1])
    assert torch.count_nonzero(raw_down[:, :, logical:]) == 0
    assert torch.equal(gs[:, :logical], source_tensors[2][:, :logical])
    assert torch.equal(us[:, :logical], source_tensors[2][:, logical:])
    assert torch.all(gs[:, logical:physical] == 1) and torch.all(us[:, logical:physical] == 1)
    assert torch.count_nonzero(gs[:, physical:]) == 0 and torch.count_nonzero(us[:, physical:]) == 0
    assert torch.equal(ds[:, :h], source_tensors[3])
    assert torch.count_nonzero(ds[:, h:]) == 0


@pytest.mark.parametrize("marker,value", [
    ("rpu_weight_mode", 5), ("rpu_int8_scale_axis", 0), ("rpu_quant_profile", "other")])
def test_int8_routed_storage_requires_exact_mode1_markers(marker, value):
    from rpu_backend.adapters.qwen3_5_moe.weights import _prepare_routed

    module = _int8_experts(2, 64, 512, torch.Generator().manual_seed(9))
    setattr(module, marker, value)
    with pytest.raises(ValueError, match="mode-1 markers"):
        _prepare_routed(module, 8)


def test_int8_layer_preparation_pads_base_scales_and_rejects_acc32():
    from rpu_backend.adapters.qwen3_5_moe.weights import prepare_moe_layer_weights

    generator = torch.Generator().manual_seed(64)
    e, h, logical = 2, 64, 512

    def fp16(*shape):
        return (torch.randn(*shape, generator=generator) * 0.05).half()
    mlp = SimpleNamespace(
        experts=_int8_experts(e, h, logical, generator), gate=SimpleNamespace(weight=fp16(256, h)),
        shared_expert=SimpleNamespace(gate_proj=SimpleNamespace(weight=fp16(logical, h)),
                                      up_proj=SimpleNamespace(weight=fp16(logical, h)),
                                      down_proj=SimpleNamespace(weight=fp16(h, logical))),
        shared_expert_gate=SimpleNamespace(weight=fp16(1, h)))
    # Routed INT8 keeps the full six-lane routed domain (UINT4 alone drops to four).
    packed = prepare_moe_layer_weights(mlp, num_cores=6, routed_num_cores=6, device="cpu")
    assert packed.routed_weight_mode == 1
    for scale, logical_width in ((packed.routed_gate_scale, 576),
                                 (packed.routed_up_scale, 576),
                                 (packed.routed_down_scale, h)):
        assert scale.shape == (e, ((logical_width + 127) // 128) * 128)
        assert torch.count_nonzero(scale[:, logical_width:]) == 0
    with pytest.raises(ValueError, match="routed expert core layout"):
        prepare_moe_layer_weights(mlp, num_cores=6, routed_num_cores=4, device="cpu")
    with pytest.raises(ValueError, match="ACC16 only"):
        prepare_moe_layer_weights(mlp, num_cores=8, device="cpu", grouped_acc32=True)


def test_int8_topology_and_accumulation_follow_the_profile():
    from rpu_backend.adapters.qwen3_5_moe.cores import MoeTopology
    from rpu_backend.adapters.qwen3_5_moe.text import _validate_qwen3_5_moe_linear_accumulation

    assert MoeTopology(6, 4, 6, 4, routed_weight_mode=1).routed_tp == 6
    assert MoeTopology(6, 4, 6, 4, routed_weight_mode=6).routed_tp == 4
    _validate_qwen3_5_moe_linear_accumulation(
        quant_scope.INT8_PROFILE_ID, {"prefill": {"linear_acc32": False}})
    with pytest.raises(Exception, match="INT8"):
        _validate_qwen3_5_moe_linear_accumulation(
            quant_scope.INT8_PROFILE_ID, {"prefill": {"linear_acc32": True}})


def test_native_admits_mode1_with_signed_storage_and_keeps_resident_closed():
    source_text = NATIVE.read_text()
    # Python and native agree on the mode-1 tag and its signed storage.
    assert re.search(r"constexpr int64_t kRoutedInt8Mode = (\d+);", source_text).group(1) == str(
        quant_scope.runtime_capabilities(quant_scope.INT8_PROFILE_ID).routed_weight_mode)
    assert "routed_dtype = routed_int8 ? at::kChar : at::kByte" in source_text
    assert "mixers_fp16 = routed_uint4 || routed_int8" in source_text
    assert "TORCH_CHECK(!routed_int8 || !linear_acc32_" in source_text
    acc32 = source_text[source_text.index("void Qwen3_5MoeModel::set_linear_acc32"):]
    assert "kRoutedInt8Mode" in acc32[:acc32.index("linear_acc32_ = enabled")]
    resident = source_text[source_text.index("void Qwen3_5MoeModel::set_resident_prefill"):]
    resident = resident[:resident.index("for (const auto& layer")]
    assert "kRoutedInt8Mode" not in resident and "weight_mode == 1" not in resident
