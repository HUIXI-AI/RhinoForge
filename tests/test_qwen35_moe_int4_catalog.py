"""Public MoE INT4 identity and cold path checks; no weights or device required."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from _support_status import expected_support_status

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/configs/qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128.toml"
GPTQ_CONFIG = CONFIG.with_name("w4a16_experts_gptq_g128.toml")
VL_CONFIGS = ROOT / "examples/configs/qwen3_5/vl/35b_a3b"


def load(name, relative, monkeypatch):
    if relative.startswith("examples/"):
        return importlib.import_module(relative[:-3].replace("/", "."))
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    return load("_common", "examples/common/config.py", monkeypatch)


def metadata(common, source="rtn"):
    scope = common._backend_leaf("adapters/qwen3_5_moe/quant_scope.py")
    text = dict(scope._EXACT_TEXT_FIELDS)
    text.update(layer_types=["linear_attention"] * 3 + ["full_attention"])
    text["layer_types"] *= 10
    text["rope_parameters"] = dict(scope._EXACT_ROPE_FIELDS)
    result = {
        "model_type": "qwen3_5_moe", "tie_word_embeddings": False,
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": text, "vision_config": dict(scope._EXACT_VISION_FIELDS),
        "rtn_metadata": {"algorithm": "RTN", "manifest": "derivation_manifest.json", "calibration": False},
        "quantization_config": {"bits": 4, "group_size": 128, "sym": True,
                                "desc_act": False, "quant_method": "gptq"},
    }
    if source == "gptq":
        result.pop("rtn_metadata")
        result["quantization_config"].update(
            damp_percent=0.01, static_groups=False, true_sequential=True,
            dynamic={"lm_head": {}, "model.language_model.embed_tokens": {},
                     "-:.*attn.*": {}, "-:.*shared_expert.*": {},
                     "-:.*mtp.*": {}, "-:.*visual.*": {}},
            modules_to_not_convert=[])
    return result


def test_rtn_public_profile_keeps_bounded_uncertified_identity(common, monkeypatch, tmp_path):
    config = common.load_config(CONFIG)
    profile = common.planner_profiles()[config["example"]["profile_id"]]
    limits = profile["input_envelope"]["limits"]
    assert profile["support"]["status"] == expected_support_status(profile, "IMPLEMENTED_UNVERIFIED_HW")
    assert profile["hw_status"] == "IMPLEMENTED_UNVERIFIED_HW"
    assert profile["component"] == "text" and "formal_certification" not in limits
    assert limits["allowed_num_cores"] == [4, 6, 8]
    assert (limits["batch_size"], limits["group_size"]) == (1, 128)
    assert config["rpu_execution"]["prefill"]["linear_acc32"] is False
    registry = load("rtn_model_registry", "python/rpu_backend/model_registry.py", monkeypatch)
    monkeypatch.setenv("RPU_MODEL_CACHE", str(tmp_path))
    assert registry.model_path(config["example"]["registry_alias"]) == tmp_path / "Qwen3.5-35B-A3B-RTN-W4A16-G128"


def test_gptq_public_profile_keeps_bounded_uncertified_identity(common, monkeypatch, tmp_path):
    config = common.load_config(GPTQ_CONFIG)
    profile = common.planner_profiles()[config["example"]["profile_id"]]
    limits = profile["input_envelope"]["limits"]
    assert profile["support"]["status"] == expected_support_status(profile, "IMPLEMENTED_UNVERIFIED_HW")
    assert profile["hw_status"] == "IMPLEMENTED_UNVERIFIED_HW"
    assert profile["component"] == "text" and "formal_certification" not in limits
    assert limits["allowed_num_cores"] == [4, 6, 8]
    assert (limits["batch_size"], limits["group_size"]) == (1, 128)
    assert config["rpu_execution"]["prefill"]["linear_acc32"] is False
    assert "padding_rows" not in config["rpu_execution"]["prefill"]
    registry = load("gptq_model_registry", "python/rpu_backend/model_registry.py", monkeypatch)
    monkeypatch.setenv("RPU_MODEL_CACHE", str(tmp_path))
    assert registry.model_path(config["example"]["registry_alias"]) == tmp_path / "Qwen3.5-35B-A3B-GPTQ-Int4"


@pytest.mark.parametrize("config_path", [CONFIG, GPTQ_CONFIG], ids=["rtn", "gptq"])
@pytest.mark.parametrize("field,value", [("cores", 5), ("cores", 7), ("acc32", True),
                                        ("alias", "qwen3_5-35b-a3b-mixed"),
                                        ("component", "multimodal")])
def test_int4_config_rejects_wrong_core_precision_or_profile(common, field, value, config_path):
    config = common.load_config(config_path)
    profile = deepcopy(common.planner_profiles()[config["example"]["profile_id"]])
    if field == "cores":
        config["rpu_execution"]["model"]["num_cores"] = value
    elif field == "acc32":
        config["rpu_execution"]["prefill"]["linear_acc32"] = value
    else:
        profile[{"alias": "asset_alias", "component": "component"}[field]] = value
    with pytest.raises(ValueError, match="model.num_cores" if field == "cores" else "INT4"):
        common._validate_execution(config["rpu_execution"], profile=profile)


@pytest.mark.parametrize("mutation", [None, "missing_rtn", "g64", "act_order", "wrong_size"])
def test_rtn_checkpoint_path_override_still_validates_source(common, tmp_path, mutation):
    data = metadata(common)
    if mutation == "missing_rtn":
        data.pop("rtn_metadata")
    elif mutation == "g64":
        data["quantization_config"]["group_size"] = 64
    elif mutation == "act_order":
        data["quantization_config"]["desc_act"] = True
    elif mutation == "wrong_size":
        data["text_config"]["hidden_size"] = 4096
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = common.load_config(CONFIG)
    if mutation is None:
        common.validate_checkpoint_profile(config, str(tmp_path))
    else:
        with pytest.raises(common.ConfigError, match="rejects checkpoint"):
            common.validate_checkpoint_profile(config, str(tmp_path))


def test_wrong_checkpoint_fails_before_tokenizer_and_weight_loading(common, monkeypatch, tmp_path):
    runner = load("rtn_public_runner", "examples/runners/qwen3_5/runner.py", monkeypatch)
    data = metadata(common)
    data.pop("rtn_metadata")
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = common.load_config(CONFIG)
    config["_model_override"] = str(tmp_path)
    tokenizer, weights, alias = Mock(), Mock(), Mock(side_effect=AssertionError("override ignored"))
    for name, values in {
        "transformers": {"AutoTokenizer": SimpleNamespace(from_pretrained=tokenizer)},
        "rpu_backend.model_registry": {"model_path": alias},
        "rpu_backend.api": {"RPUModelForConditionalGeneration": SimpleNamespace(from_pretrained=weights)},
    }.items():
        stub = ModuleType(name)
        vars(stub).update(values)
        monkeypatch.setitem(sys.modules, name, stub)
    with pytest.raises(common.ConfigError, match="rejects checkpoint"):
        runner.prepare(config, SimpleNamespace(owner=None))
    tokenizer.assert_not_called()
    weights.assert_not_called()
    alias.assert_not_called()


@pytest.mark.parametrize("config_path", [CONFIG, GPTQ_CONFIG], ids=["rtn", "gptq"])
def test_valid_fp8_checkpoint_cannot_be_relabelled_as_int4(common, tmp_path, config_path):
    from test_qwen35_moe_python_integration import _exact_config

    (tmp_path / "config.json").write_text(json.dumps(_exact_config()))
    with pytest.raises(common.ConfigError, match="requires checkpoint profile"):
        common.validate_checkpoint_profile(common.load_config(config_path), str(tmp_path))


@pytest.mark.parametrize("config_path,source", [(CONFIG, "gptq"), (GPTQ_CONFIG, "rtn")])
def test_rtn_and_official_gptq_sources_are_not_interchangeable(common, tmp_path, config_path, source):
    (tmp_path / "config.json").write_text(json.dumps(metadata(common, source)))
    config = common.load_config(config_path)
    config["_model_override"] = str(tmp_path)
    with pytest.raises(common.ConfigError, match="requires checkpoint profile"):
        common.validate_checkpoint_profile(config, str(tmp_path))


def test_fp8_path_override_cannot_be_relabelled_with_rtn_weights(common, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(metadata(common)))
    config = common.load_config(CONFIG.with_name("fp8_e4m3_experts_attn_gdn.toml"))
    config["_model_override"] = str(tmp_path)
    with pytest.raises(common.ConfigError, match="requires checkpoint profile"):
        common.validate_checkpoint_profile(config, str(tmp_path))


@pytest.mark.parametrize("config_path,source", [(CONFIG, "rtn"), (GPTQ_CONFIG, "gptq")])
def test_int4_override_reaches_moe_loader_and_cache(common, monkeypatch, tmp_path, config_path, source):
    import torch

    runner = load("rtn_public_success_runner", "examples/runners/qwen3_5/runner.py", monkeypatch)
    (tmp_path / "config.json").write_text(json.dumps(metadata(common, source)))
    config = common.load_config(config_path)
    config["_model_override"] = str(tmp_path)
    tokenizer = Mock(return_value=SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]])))
    tokenizer_load = Mock(return_value=tokenizer)
    model = SimpleNamespace(config=SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"]))
    loader = Mock(return_value=model)
    install = Mock(return_value=model)
    adapter = Mock(return_value=SimpleNamespace(to_rpu=install))
    cache = Mock(return_value=object())
    generate = Mock()
    monkeypatch.setattr(importlib.import_module("examples.common.generation"), "_generate", generate)
    for name, values in {
        "transformers": {"AutoTokenizer": SimpleNamespace(from_pretrained=tokenizer_load)},
        "rpu_backend.model_registry": {"model_path": Mock(side_effect=AssertionError("override ignored"))},
        "rpu_backend.api": {"RPUModelForConditionalGeneration": SimpleNamespace(from_pretrained=loader),
                            "Qwen3_5MoeCache": SimpleNamespace(from_model=cache)},
        "rpu_backend.adapters.qwen3_5_moe": {"Qwen3_5MoeAdapter": adapter},
    }.items():
        stub = ModuleType(name)
        vars(stub).update(values)
        monkeypatch.setitem(sys.modules, name, stub)
    infer, owner = runner.prepare(config, SimpleNamespace(owner=None))
    assert owner is model
    infer()
    tokenizer_load.assert_called_once_with(str(tmp_path), local_files_only=True)
    loader.assert_called_once_with(str(tmp_path), dtype=torch.float16, local_files_only=True,
                                   rpu_execution=config["rpu_execution"])
    adapter.assert_called_once_with(model, rpu_execution=config["rpu_execution"])
    install.assert_called_once_with(max_seq_len=34)
    cache.assert_called_once_with(model, max_seq_len=34)
    assert generate.call_args.kwargs["token_selector"] is None


def test_model_cli_override_is_recorded_without_changing_registered_identity(common, monkeypatch, tmp_path):
    from contextlib import nullcontext
    from examples.common import cli, measurement, logging
    monkeypatch.setattr(logging, "logging_session", lambda *_a, **_kw: nullcontext())
    captured = {}
    monkeypatch.setattr(measurement, "run", lambda config: captured.update(config) or 0)
    for path, alias in ((CONFIG, "qwen3_5-35b-a3b-rtn-w4a16"),
                        (VL_CONFIGS / "w4a16_experts_gptq_g128.toml", "qwen3_5-35b-a3b-gptq-w4a16")):
        assert cli.main(["--config", str(path), "--model", str(tmp_path)]) == 0
        assert captured["_model_override"] == str(tmp_path.resolve())
        assert captured["example"]["registry_alias"] == alias
    for path in (ROOT / "examples/configs/qwen3/text/0_6b/fp16.toml",
                 ROOT / "examples/configs/qwen3_5/vl/2b/fp16.toml",
                 ROOT / "examples/configs/qwen3_5/text/0_8b/fp16.toml"):
        with pytest.raises(SystemExit) as error:
            cli.main(["--config", str(path), "--model", str(tmp_path)])
        assert error.value.code == 2


@pytest.mark.parametrize("source", ["rtn", "gptq"])
def test_int4_image_profiles_keep_bounded_uncertified_identity(common, source):
    config = common.load_config(VL_CONFIGS / f"w4a16_experts_{source}_g128.toml")
    profile = common.planner_profiles()[config["example"]["profile_id"]]
    limits = profile["input_envelope"]["limits"]
    assert profile["owner"] == profile["authority"] == "composite-qwen35-moe"
    assert profile["hw_status"] == "IMPLEMENTED_UNVERIFIED_HW"
    assert profile["support"]["status"] == expected_support_status(profile, "IMPLEMENTED_UNVERIFIED_HW")
    assert profile["component"] == "multimodal" and "formal_certification" not in limits
    assert limits["allowed_num_cores"] == [4, 6, 8]
    assert (limits["batch_size"], limits["group_size"]) == (1, 128)
    assert limits["video_supported"] is False
    assert limits["component_paths"] == ["language_model", "vision_encoder"]
    assert config["rpu_execution"]["prefill"]["linear_acc32"] is False
    # Vision's cold accumulation is independent of the routed INT4 text path.
    config["rpu_execution"]["vision"]["linear_acc32"] = True
    common._validate_execution(config["rpu_execution"], profile=profile)
    config["rpu_execution"]["prefill"]["linear_acc32"] = True
    with pytest.raises(common.ConfigError, match="ACC16"):
        common._validate_execution(config["rpu_execution"], profile=profile)
    config["rpu_execution"]["prefill"]["linear_acc32"] = False
    for count in (4, 6, 8):
        config["rpu_execution"]["model"]["num_cores"] = count
        common._validate_execution(config["rpu_execution"], profile=profile)
    config["input"]["videos"] = ["clip.mp4"]
    with pytest.raises(common.ConfigError, match="video"):
        common.validate_generation_options(config)


@pytest.mark.parametrize("source", ["rtn", "gptq"])
@pytest.mark.parametrize("wrong_source", [False, True])
def test_int4_image_override_binds_source_before_processor_and_weights(
        common, monkeypatch, tmp_path, source, wrong_source):
    import torch

    data_source = ("gptq" if source == "rtn" else "rtn") if wrong_source else source
    (tmp_path / "config.json").write_text(json.dumps(metadata(common, data_source)))
    config = common.load_config(VL_CONFIGS / f"w4a16_experts_{source}_g128.toml")
    config["_model_override"] = str(tmp_path)
    runner = load("int4_controlled_runner", "examples/runners/qwen3_5/vision.py", monkeypatch)
    text_runner = load("int4_generation", "examples/common/generation.py", monkeypatch)
    generate = Mock()
    monkeypatch.setattr(text_runner, "_generate", generate)
    images = Mock(return_value=[object()])
    monkeypatch.setattr(runner, "_images", images)
    encoded = {"input_ids": torch.ones((1, 65), dtype=torch.long),
               "pixel_values": object(), "image_grid_thw": object(),
               "mm_token_type_ids": object(), "attention_mask": object()}
    processor = Mock(return_value=encoded)
    processor_load = Mock(return_value=processor)
    model = object()
    loader = Mock(return_value=model)
    install = Mock(return_value=model)
    adapter = Mock(return_value=SimpleNamespace(to_rpu=install))
    cache = Mock(return_value=object())
    for name, values in {
        "transformers": {"AutoProcessor": SimpleNamespace(from_pretrained=processor_load)},
        "rpu_backend.api": {"RPUModelForConditionalGeneration": SimpleNamespace(from_pretrained=loader)},
        "rpu_backend.adapters.qwen3_5_moe": {"Qwen3_5MoeAdapter": adapter},
        "rpu_backend.api.qwen3_5_moe_cache": {"Qwen3_5MoeCache": SimpleNamespace(from_model=cache)},
    }.items():
        stub = ModuleType(name)
        vars(stub).update(values)
        monkeypatch.setitem(sys.modules, name, stub)
    request = SimpleNamespace(source_config=config, owner=None)
    if wrong_source:
        with pytest.raises(common.ConfigError, match="requires checkpoint profile"):
            runner._qwen35_vision(request)
        processor_load.assert_not_called()
        loader.assert_not_called()
        images.assert_not_called()
        assert request.owner is None
        return
    infer = runner._qwen35_vision(request)
    infer()
    assert request.owner is model
    processor_load.assert_called_once_with(str(tmp_path), local_files_only=True)
    loader.assert_called_once_with(str(tmp_path), dtype=torch.float16, local_files_only=True,
                                   rpu_execution=config["rpu_execution"])
    adapter.assert_called_once_with(model, rpu_execution=config["rpu_execution"])
    install.assert_called_once_with(max_seq_len=128)
    cache.assert_called_once_with(model, max_seq_len=128)
    assert generate.call_args.args[4]["pixel_values"] is encoded["pixel_values"]
    assert generate.call_args.args[4]["mm_token_type_ids"] is encoded["mm_token_type_ids"]


def _prose_config(common, stem, length):
    config = common.load_config(CONFIG.with_name(f"{stem}.toml"))
    config["input"] = {
        "prompt_file": str(CONFIG.with_name("prefill_realtext.txt")),
        "prompt_sha256": "1592273bd09459f92433dd66fc4c6bb49854f922c78ce1cf9cd785f730fd567b",
        "input_ids_sha256": {
            64: "84480760da4e674d12a5fd4bf70355a3155f1ef0811de627f866833b44673a8a",
            4096: "8628855580e3c9e42423ef9500cd85c2b94fe95f83a3de991caebe93affe0ad1",
        }[length],
        "prefill_tokens": length, "decode_steps": 4096, "stop_on_eos": False,
    }
    config["run"].update(inference_mode=False, timing_mode="forward_argmax")
    config["cpu"] = {"threads": 24}
    # The published prose numbers were measured on the ordinary dataflow, so pin
    # resident off here instead of inheriting whatever the template now defaults to.
    config["rpu_execution"]["prefill"].update(padding_rows=0, linear_acc32=False,
                                              moe_resident_prefill=False)
    common.validate_generation_options(config)
    return config


def test_moe_performance_parameters_share_prose_and_reject_synthetic_input(common):
    seen = {64: set(), 4096: set()}
    prose = set()
    for stem in ("fp8_e4m3_experts_attn_gdn", "w4a16_experts_rtn_g128", "w4a16_experts_gptq_g128"):
        for length in (64, 4096):
            config = _prose_config(common, stem, length)
            seen[length].add(config["input"]["input_ids_sha256"])
            prose.add((config["input"]["prompt_file"], config["input"]["prompt_sha256"]))
            assert config["input"]["prefill_tokens"] == length
            assert config["input"]["decode_steps"] == 4096
            assert config["input"]["stop_on_eos"] is False
            assert config["run"]["warmup"] == 2 and config["run"]["runs"] == 5
            assert config["run"].get("warmup_decode_steps", config["input"]["decode_steps"]) == 4096
            assert config["cpu"]["threads"] == 24
            assert config["run"]["inference_mode"] is False
            assert config["run"]["timing_mode"] == "forward_argmax"
            assert config["rpu_execution"]["model"]["num_cores"] == 8
            assert config["rpu_execution"]["prefill"] == {
                "chunk_size": "auto", "padding_rows": 0, "linear_acc32": False,
                "moe_resident_prefill": False}
    assert len(prose) == 1 and all(len(hashes) == 1 for hashes in seen.values())
    # The previous EOS-fill contract cannot be restored under the same MoE label.
    config["input"]["prefill_fill_token_id"] = 248046
    with pytest.raises(common.ConfigError, match="synthetic"):
        common.validate_generation_options(config)
    del config["input"]["prefill_fill_token_id"]
    config["example"] = common.load_config(
        ROOT / "examples/configs/qwen3_5/text/2b/fp16.toml")["example"]
    with pytest.raises(common.ConfigError, match="MoE text"):
        common.validate_generation_options(config)


def test_short_moe_prose_fails_before_loading_weights(common, monkeypatch, tmp_path):
    import torch

    config = _prose_config(common, "w4a16_experts_rtn_g128", 64)
    (tmp_path / "config.json").write_text(json.dumps(metadata(common)))
    config["_model_override"] = str(tmp_path)
    runner = load("short_moe_prose_runner", "examples/runners/qwen3_5/runner.py", monkeypatch)
    tokenizer = Mock(return_value={"input_ids": torch.arange(63).reshape(1, -1)})
    weights = Mock()
    for name, values in {
        "transformers": {"AutoTokenizer": SimpleNamespace(from_pretrained=lambda *_a, **_k: tokenizer)},
        "rpu_backend.model_registry": {"model_path": Mock(side_effect=AssertionError("override ignored"))},
        "rpu_backend.api": {"RPUModelForConditionalGeneration": SimpleNamespace(from_pretrained=weights)},
    }.items():
        stub = ModuleType(name)
        vars(stub).update(values)
        monkeypatch.setitem(sys.modules, name, stub)
    owner = SimpleNamespace(owner=None)
    with pytest.raises(ValueError, match="fewer than requested 64"):
        runner.prepare(config, owner)
    weights.assert_not_called()
    assert owner.owner is None
