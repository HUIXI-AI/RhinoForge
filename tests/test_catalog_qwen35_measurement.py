"""Host regressions for the published Dense VL input and timer boundaries."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    import _common
    import _qwen35_vision
    import _runner
    import _text_vision_runners
    return SimpleNamespace(common=_common, controlled=_qwen35_vision,
                           shared=_runner, text=_text_vision_runners)


def legacy_config():
    return {
        "example": {"profile_id": "qwen3_5-2b.fp16.vision-controlled",
                    "registry_alias": "qwen3_5-2b", "target": "qwen3_5_vision",
                    "architecture": "Qwen3_5ForConditionalGeneration", "dtype": "fp16",
                    "opt_in": {"QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED": "1"}},
        "input": {"images": ["image.png"], "image_size": 448,
                  "prompt": "Describe the image in detail. compiler paragraph",
                  "prefill_filler_prompt": "compiler paragraph", "prefill_tokens": 512,
                  "decode_steps": 192, "stop_on_eos": False},
        "run": {"timing_mode": "legacy_dense_vl", "summary_statistic": "median",
                "warmup": 1, "runs": 3, "torch_num_threads": 12, "inference_mode": True},
        "rpu_execution": {"model": {"num_cores": 8}, "prefill": {"chunk_size": "auto"}},
    }


@pytest.mark.parametrize("change,match", [
    ({"example": {"architecture": "Qwen3_5MoeForConditionalGeneration"}}, "Dense FP16"),
    ({"example": {"target": "qwen3_5"}}, "image_size"),
    ({"example": {"dtype": "w8a16"}}, "Dense FP16"),
    ({"run": {"timing_mode": "loop"}}, "timing_mode"),
    ({"run": {"summary_statistic": "minimum"}}, "summary_statistic"),
    ({"run": {"warmup_decode_steps": 3}}, "full decode warmup"),
    ({"run": {"inference_mode": False}}, "inference_mode"),
    ({"input": {"stop_on_eos": True}}, "fixed prefill_tokens"),
    ({"input": {"image_size": True}}, "image_size"),
    ({"input": {"prefill_fill_token_id": 99}}, "no token-ID filler"),
])
def test_unsupported_options_fail_before_model_or_processor_load(modules, monkeypatch, change, match):
    config = legacy_config()
    for section, values in change.items():
        config[section].update(values)
    # The public controlled entry performs this check before backend imports,
    # checkpoint lookup, processor construction or model allocation.
    checkpoint = Mock(side_effect=AssertionError("model preparation must not begin"))
    monkeypatch.setattr(modules.controlled, "_checkpoint", checkpoint)
    with pytest.raises(modules.common.ConfigError, match=match):
        modules.controlled._qwen35_vision(SimpleNamespace(source_config=config))
    checkpoint.assert_not_called()


def test_dense_vl_fixed_lengths_pass_public_config_validation(modules, tmp_path):
    text = '''[example]
profile_id = "qwen3_5-2b.fp16.vision-controlled"
registry_alias = "qwen3_5-2b"
target = "qwen3_5_vision"
architecture = "Qwen3_5ForConditionalGeneration"
dtype = "fp16"
[example.opt_in]
QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED = "1"
[input]
images = ["image.png"]
image_size = 448
prompt = "image prompt"
prefill_filler_prompt = "compiler paragraph"
prefill_tokens = 512
decode_steps = 192
stop_on_eos = false
[run]
timing_mode = "legacy_dense_vl"
summary_statistic = "median"
warmup_decode_steps = 192
inference_mode = true
'''
    path = tmp_path / "vl.toml"
    path.write_text(text)
    assert modules.common.load_config(path)["input"]["prefill_tokens"] == 512


def test_generation_statistic_rejects_encoder_only_config(modules, tmp_path):
    source = ROOT / "examples/configs/qwen3_vl/vision/8b/fp16.toml"
    path = tmp_path / "encoder.toml"
    path.write_text(source.read_text().replace(
        "[run]", '[run]\nsummary_statistic = "median"'))
    with pytest.raises(modules.common.ConfigError, match="text-generation"):
        modules.common.load_config(path)


def test_rgb_resize_and_explicit_filler_preserve_multimodal_chat_boundary(modules, tmp_path):
    from PIL import Image

    path = tmp_path / "image.png"
    Image.new("L", (21, 17), color=93).save(path)
    images = modules.controlled._images({"images": [str(path)], "image_size": 448})
    assert images[0].mode == "RGB" and images[0].size == (448, 448)
    assert images[0].getpixel((0, 0)) == (93, 93, 93)
    tokenizer = Mock(return_value=SimpleNamespace(input_ids=torch.tensor([[10, 11]])))
    tokenizer.convert_tokens_to_ids.return_value = 91
    encoded = {"input_ids": torch.tensor([[80, 91, 80, 90, 12, 91, 92]]),
               "mm_token_type_ids": torch.tensor([[0, 0, 0, 1, 0, 0, 0]]),
               "attention_mask": torch.ones((1, 7), dtype=torch.long),
               "pixel_values": object(), "image_grid_thw": object()}
    pixels, grid = encoded["pixel_values"], encoded["image_grid_thw"]
    result = modules.text._extend_multimodal_input(encoded, {
        "prompt": "different image prompt", "prefill_filler_prompt": "compiler paragraph",
        "prefill_tokens": 10}, tokenizer)
    tokenizer.assert_called_once_with("compiler paragraph", add_special_tokens=False, return_tensors="pt")
    assert result["input_ids"].tolist() == [[80, 91, 80, 90, 12, 10, 11, 10, 91, 92]]
    assert result["mm_token_type_ids"].tolist() == [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]
    assert result["attention_mask"].tolist() == [[1] * 10]
    assert result["pixel_values"] is pixels and result["image_grid_thw"] is grid


def test_legacy_loop_includes_token_snapshots_and_boundary_checks_only(modules, monkeypatch):
    clock = [0.]
    buffer = torch.zeros((1, 1, 4))
    sequence = iter([2, 1, 3])
    calls = []

    def forward(**kwargs):
        calls.append(kwargs)
        clock[0] += .100
        buffer.zero_()
        buffer[0, 0, next(sequence)] = 5
        return SimpleNamespace(logits=buffer)

    model = Mock(side_effect=forward)
    model.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=4))
    original_argmax, original_clone = torch.Tensor.argmax, torch.Tensor.clone
    original_check = modules.text._legacy_checked_logits
    checks = []

    def argmax(tensor, *args, **kwargs):
        clock[0] += .020
        return original_argmax(tensor, *args, **kwargs)

    def clone(tensor, *args, **kwargs):
        if tensor.dtype == torch.long and tensor.shape == (1, 1):
            clock[0] += .007
        return original_clone(tensor, *args, **kwargs)

    def checked(output, vocab):
        checks.append(len(calls))
        clock[0] += .040
        return original_check(output, vocab)

    monkeypatch.setattr(torch.Tensor, "argmax", argmax)
    monkeypatch.setattr(torch.Tensor, "clone", clone)
    monkeypatch.setattr(modules.text, "_legacy_checked_logits", checked)
    monkeypatch.setattr(modules.text.time, "perf_counter", lambda: clock[0])
    cache = Mock()
    result = modules.text._generate(model, cache, torch.tensor([[1, 2, 3]]), 2,
                                    stop_on_eos=False, timing_mode="legacy_dense_vl")
    assert result["generation"]["prefill_ms"] == pytest.approx(127.)
    assert result["generation"]["decode_ms"] == pytest.approx(334.)
    assert checks == [1, 2, 3, 3]
    assert all(call["logits_to_keep"] == 1 for call in calls)
    assert [call["input_ids"].tolist() for call in calls] == [[[1, 2, 3]], [[2]], [[1]]]
    assert result["token_ids"].tolist() == [[2, 1, 3]]
    buffer.fill_(0)
    assert result["first_logits"].tolist() == [[[0., 0., 5., 0.]]]
    assert result["logits"].tolist() == [[[0., 0., 0., 5.]]]
    cache.reset.assert_called_once_with()


@pytest.mark.parametrize("logits", [torch.zeros((1, 4)), torch.full((1, 1, 4), float("nan"))])
def test_legacy_logits_boundary_rejects_bad_shape_or_nonfinite(modules, logits):
    with pytest.raises(ValueError, match="finite with shape"):
        modules.text._legacy_checked_logits(SimpleNamespace(logits=logits), 4)


def test_median_rates_are_not_rates_from_mean_time_and_default_is_unchanged(modules):
    samples = [{"prefill_tokens": 100, "decode_calls": 10,
                "prefill_ms": ms, "decode_ms": ms,
                "prefill_tokens_per_second": 100000 / ms,
                "decode_tokens_per_second": 10000 / ms} for ms in (100., 200., 900.)]
    median = modules.shared._generation_summary(samples, "median")
    assert median["prefill_ms_median"] == median["decode_ms_median"] == 200.
    assert median["prefill_tokens_per_second"] == 500.
    assert median["decode_tokens_per_second"] == 50.
    mean = modules.shared._generation_summary(samples)
    assert mean["prefill_ms_mean"] == mean["decode_ms_mean"] == 400.
    assert mean["prefill_tokens_per_second"] == 250.
    assert mean["decode_tokens_per_second"] == 25.


@pytest.mark.parametrize("cache_failure", [False, True])
def test_legacy_setup_binds_one_execution_and_exact_capacity_before_upload(
        modules, monkeypatch, cache_failure):
    config = legacy_config()
    model, raw, adapter = Mock(), Mock(), Mock()
    raw.half.return_value = raw
    adapter.to_rpu.return_value = model
    constructor, adapter_constructor = Mock(return_value=raw), Mock(return_value=adapter)
    cache = Mock()
    cache_constructor = Mock(return_value=cache)
    if cache_failure:
        cache_constructor.side_effect = RuntimeError("cache setup failed")

    def install_module(name, **values):
        module = ModuleType(name)
        vars(module).update(values)
        monkeypatch.setitem(sys.modules, name, module)

    install_module("rpu_backend.api", RPUModelForConditionalGeneration=SimpleNamespace(from_pretrained=constructor))
    install_module("rpu_backend.adapters.qwen3_5", Qwen3_5Adapter=adapter_constructor)
    install_module("rpu_backend.api.qwen3_5_cache", Qwen3_5Cache=SimpleNamespace(from_model=cache_constructor))
    processor = Mock(return_value={
        "input_ids": torch.ones((1, 512), dtype=torch.long),
        "pixel_values": torch.ones((1, 8)),
        "image_grid_thw": torch.tensor([[1, 16, 16]]),
        "mm_token_type_ids": torch.zeros((1, 512), dtype=torch.long),
    })
    metadata = SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=SimpleNamespace(layer_types=["linear_attention", "full_attention"]), quant_config=None)
    install_module("transformers", AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
                   AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=metadata)))
    monkeypatch.setattr(modules.controlled, "_images", lambda _: [object()])
    monkeypatch.setattr(modules.controlled, "_checkpoint", lambda _: Path("/model"))
    original_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            args = ("cpu", *args[1:])
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    generate = Mock(return_value={"text": "generated"})
    monkeypatch.setattr(modules.text, "_generate", generate)
    if cache_failure:
        with pytest.raises(RuntimeError, match="cache setup failed"):
            modules.controlled.prepare_vision(config)
        model.close.assert_called_once_with()
        generate.assert_not_called()
    else:
        infer, owner = modules.controlled.prepare_vision(config)
        assert owner is model and infer() == {"text": "generated"}
        assert generate.call_args.kwargs["timing_mode"] == "legacy_dense_vl"
        assert generate.call_args.kwargs["stop_on_eos"] is False
        assert generate.call_args.args[3] == 192
        assert "attention_mask" not in generate.call_args.args[4]
    assert "device" not in constructor.call_args.kwargs
    assert constructor.call_args.kwargs["rpu_execution"] is config["rpu_execution"]
    assert adapter_constructor.call_args.kwargs["rpu_execution"] is config["rpu_execution"]
    assert adapter_constructor.call_args.args == (raw,)
    raw.half.assert_called_once_with()
    adapter.to_rpu.assert_called_once_with(max_seq_len=704)
    cache_constructor.assert_called_once_with(model, max_seq_len=704)


@pytest.mark.parametrize("architecture,quant,match", [
    ("Qwen3_5MoeForConditionalGeneration", None, "architecture"),
    ("Qwen3_5ForConditionalGeneration", {"method": "w8a16"}, "precision"),
])
def test_qwen35_vl_checkpoint_label_mismatch_precedes_images_processor_and_weights(
        modules, monkeypatch, architecture, quant, match):
    config = legacy_config()
    metadata = SimpleNamespace(architectures=[architecture],
        text_config=SimpleNamespace(layer_types=["linear_attention", "full_attention"]), quant_config=quant)
    processor, loader, images = Mock(), Mock(), Mock()
    for name, values in {
        "transformers": {"AutoConfig": SimpleNamespace(from_pretrained=Mock(return_value=metadata)),
                         "AutoProcessor": SimpleNamespace(from_pretrained=processor)},
        "rpu_backend.api": {"RPUModelForConditionalGeneration": SimpleNamespace(from_pretrained=loader)},
    }.items():
        module = ModuleType(name)
        vars(module).update(values)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(modules.controlled, "_checkpoint", lambda _: Path("/caller/model"))
    monkeypatch.setattr(modules.controlled, "_images", images)
    with pytest.raises(modules.common.ConfigError, match=match):
        modules.controlled._qwen35_vision(SimpleNamespace(source_config=config, owner=None))
    images.assert_not_called()
    processor.assert_not_called()
    loader.assert_not_called()
