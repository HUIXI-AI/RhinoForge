"""CPU checks for shared example measurement; never initialize the backend."""
import contextlib
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import subprocess
import runpy
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from rpu_backend.api.cache import RPUCache
from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache
from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache
from test_host_argmax_cpu import native_cpu_argmax


ROOT = Path(__file__).resolve().parents[1]
CPU = SimpleNamespace(**vars(importlib.import_module("examples.common.cpu")))


def load(name, relative):
    if relative.startswith("examples/"):
        return importlib.import_module(relative[:-3].replace("/", "."))
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def greedy_api(monkeypatch):
    generation = load("example_generation_test", "python/rpu_backend/api/generation.py")
    api = ModuleType("rpu_backend.api")
    api.greedy_token_ids = Mock(wraps=generation.greedy_token_ids)
    monkeypatch.setitem(sys.modules, "rpu_backend", ModuleType("rpu_backend"))
    monkeypatch.setitem(sys.modules, "rpu_backend.api", api)
    return api


def _prepare_public_llama_infer(monkeypatch, runner, model, cache, input_ids,
                               decode_steps, stop_on_eos):
    """Exercise the actual public prepare/infer route; only loader/tokenizer are doubles."""
    registry = ModuleType("rpu_backend.model_registry")
    registry.model_path = lambda _: "/cache/llama-test"
    tokenizer = Mock(return_value=SimpleNamespace(input_ids=input_ids))
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *_a, **_kw: tokenizer)
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setitem(sys.modules, transformers.__name__, transformers)
    backend = sys.modules["rpu_backend"]
    loader = Mock(return_value=model)
    monkeypatch.setattr(backend, "RPUModelForCausalLM",
                        SimpleNamespace(from_pretrained=loader), raising=False)
    monkeypatch.setattr(backend, "RPUCache", SimpleNamespace(
        from_model=lambda *_a, **_kw: cache), raising=False)
    state = SimpleNamespace(owner=None)
    from examples.common import config as _common
    monkeypatch.setattr(_common, "install_quantization", lambda *_: None)
    from examples.runners.llama import runner as llama_runner
    infer, owner = llama_runner.prepare({
        "example": {"target": "llama", "registry_alias": "llama-test", "profile_id": "llama-test"},
        "input": {"prompt": "unchanged request", "decode_steps": decode_steps,
                  "stop_on_eos": stop_on_eos}}, state)
    assert owner is state.owner is model
    assert loader.call_args.kwargs["dtype"] == torch.float16
    assert loader.call_args.kwargs["device"] == "rpu"
    return infer


def test_generation_reuses_cache_but_resets_each_request(greedy_api):
    runner = load("example_text_test", "examples/common/generation.py")
    cache = Mock()
    model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[[0., 2., 1.]]])))
    for _ in range(2):
        out = runner._generate(model, cache, torch.tensor([[0, 2]]), 3)
        assert out["token_ids"].tolist() == [[1, 1, 1, 1]]
    assert cache.reset.call_count == 2
    assert model.call_count == 8
    assert greedy_api.greedy_token_ids.call_count == 8
    assert all(call.kwargs["past_key_values"] is cache for call in model.call_args_list)
    for index, call in enumerate(model.call_args_list):
        if index % 4 == 0:
            assert call.kwargs["logits_to_keep"] == 1
            assert call.kwargs["input_ids"].shape == (1, 2)
        else:
            assert "logits_to_keep" not in call.kwargs
            assert call.kwargs["input_ids"].shape == (1, 1)


def _generation_cache(cache_type):
    cache = object.__new__(cache_type)
    cache.position = cache._seen_tokens = 19
    cache.max_seq_len = 256
    cache._reset_groups = None
    empty = torch.empty(0)
    cache.k_caches = [empty, torch.ones(128)]
    cache.v_caches = [empty, torch.ones(128)]
    if issubclass(cache_type, Qwen3_5Cache):
        cache.recurrent_states = [torch.ones(128), empty]
        cache.conv_states = [torch.ones(128), empty]
        cache.gdn_states = cache.recurrent_states
    return cache


@pytest.mark.parametrize("mode", ["per_call", "legacy_dense_vl"])
@pytest.mark.parametrize("cache_type,architecture", [
    (Qwen3_5Cache, "Qwen3_5ForConditionalGeneration"),
    (Qwen3_5MoeCache, "Qwen3_5MoeForConditionalGeneration"),
])
@pytest.mark.parametrize("tokens,decode_steps,skip_kv", [
    (1, 1, False), (2, 1, True), (13, 51, True), (13, 52, False),
    (64, 1, False), (129, 31, True), (129, 64, False),
])
def test_hybrid_generation_resets_kv_and_preserves_prefill_gdn_versions(
        greedy_api, mode, cache_type, architecture, tokens, decode_steps, skip_kv):
    runner = load("example_hybrid_reset_test", "examples/common/generation.py")
    cache = _generation_cache(cache_type)
    gdn = cache.recurrent_states + cache.conv_states
    tensors = cache.k_caches + cache.v_caches + gdn
    pointers = [tensor.data_ptr() for tensor in tensors]
    calls = []

    def forward(**kwargs):
        if not calls or len(calls) % (decode_steps + 1) == 0:
            assert cache.position == cache._seen_tokens == 0
            assert all(torch.all(tensor == (dirty[0] if skip_kv else 0))
                       for tensor in cache.k_caches + cache.v_caches)
            assert all(torch.all(tensor == (dirty[0] if tokens > 1 else 0))
                       for tensor in gdn if tensor.numel())
        calls.append(kwargs["input_ids"].shape[1])
        cache.update_position(kwargs["input_ids"].shape[1])
        return SimpleNamespace(logits=torch.tensor([[[0., 2., 1.]]]))

    model = Mock(side_effect=forward)
    model.config = SimpleNamespace(architectures=[architecture],
                                   text_config=SimpleNamespace(vocab_size=3))
    model._rpu_g05_vlm_ready = False
    model.generation_config = None
    tokenizer = SimpleNamespace(decode=lambda *_a, **_kw: "token")
    dirty = [0]
    for value in (7., -3., 7.):
        dirty[0] = value
        before_dirty = [tensor._version for tensor in tensors]
        for tensor in tensors:
            tensor.numpy().fill(value)  # Raw CPU alias does not bump Torch versions.
        assert [tensor._version for tensor in tensors] == before_dirty
        versions = [tensor._version for tensor in tensors]
        result = runner._generate(
            model, cache, torch.ones((1, tokens), dtype=torch.long), decode_steps,
            tokenizer=tokenizer, stop_on_eos=False, timing_mode=mode,
            input_metadata={"prefill_tokens": 4096, "decode_steps": 1,
                            "prompt_file": "not-the-input"})
        assert result["generation"]["prefill_tokens"] == tokens
        assert result["generation"]["decode_calls"] == decode_steps
        assert result["token_ids"].tolist() == [[1] * (decode_steps + 1)]
        touched = ([] if skip_kv else cache.k_caches + cache.v_caches)
        touched += gdn if tokens == 1 else []
        for tensor, version in zip(tensors, versions):
            assert tensor._version == version + sum(other is tensor for other in touched)
        assert [tensor.data_ptr() for tensor in tensors] == pointers
        # Public reset remains a complete zeroing operation, even after this route.
        cache.reset()
        assert all(not torch.count_nonzero(tensor) for tensor in tensors)
    assert calls == ([tokens] + [1] * decode_steps) * 3


@pytest.mark.parametrize("mode", ["per_call", "legacy_dense_vl"])
@pytest.mark.parametrize("kind", ["qwen3", "llama", "g05", "vision", "mismatch", "subclass"])
def test_generation_reset_keeps_other_model_and_cache_contracts(greedy_api, mode, kind):
    runner = load("example_other_reset_test", "examples/common/generation.py")
    class OtherHybrid(Qwen3_5Cache):
        pass
    cache_type = (RPUCache if kind in {"qwen3", "llama", "vision"}
                  else OtherHybrid if kind == "subclass" else Qwen3_5Cache)
    cache = _generation_cache(cache_type)
    gdn = getattr(cache, "recurrent_states", []) + getattr(cache, "conv_states", [])
    original_reset = cache.reset
    cache.reset = Mock(wraps=original_reset)

    def forward(**_kwargs):
        assert all(not torch.count_nonzero(tensor)
                   for tensor in cache.k_caches + cache.v_caches + gdn)
        return SimpleNamespace(logits=torch.tensor([[[0., 2., 1.]]]))

    model = Mock(side_effect=forward)
    architecture = {"qwen3": "Qwen3ForCausalLM", "llama": "LlamaForCausalLM",
                    "vision": "Qwen3_5VisionModel", "mismatch": "Qwen3_5MoeForConditionalGeneration"}
    model.config = SimpleNamespace(
        architectures=[architecture.get(kind, "Qwen3_5ForConditionalGeneration")],
        text_config=SimpleNamespace(vocab_size=3))
    model._rpu_g05_vlm_ready = kind == "g05"
    model.generation_config = None
    runner._generate(model, cache, torch.tensor([[1, 2]]), 1,
                     tokenizer=SimpleNamespace(decode=lambda *_a, **_kw: "token"),
                     stop_on_eos=False, timing_mode=mode)
    cache.reset.assert_called_once_with()


@pytest.mark.parametrize("target", ["qwen3", "llama", "qwen3_5_moe", "qwen3_5_dense"])
def test_plain_text_prepare_keeps_cpu_prompt_and_generated_ids(
        monkeypatch, greedy_api, target):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    from examples.common import config as _common
    monkeypatch.setattr(_common, "validate_checkpoint_profile", lambda *_: None)
    monkeypatch.setattr(_common, "install_quantization", lambda *_: None)
    runner = load("example_plain_cpu_ids", "examples/common/generation.py")
    ids = torch.tensor([[1, 2, 3]])
    cache = Mock()
    model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[[0., 2., 1.]]])))
    model.config = SimpleNamespace(eos_token_id=None)
    model.generation_config = None
    loader = Mock(return_value=model)
    backend = sys.modules["rpu_backend"]
    monkeypatch.setattr(backend, "RPUModelForCausalLM",
                        SimpleNamespace(from_pretrained=loader), raising=False)
    monkeypatch.setattr(backend, "RPUCache", SimpleNamespace(
        from_model=lambda *_a, **_kw: cache), raising=False)
    if target in {"qwen3_5_moe", "qwen3_5_dense"}:
        is_moe = target == "qwen3_5_moe"
        architecture = "Qwen3_5Moe" if is_moe else "Qwen3_5"
        adapter_name = "qwen3_5_moe" if is_moe else "qwen3_5"
        model.config.architectures = [architecture + "ForConditionalGeneration"]
        api = sys.modules["rpu_backend.api"]
        monkeypatch.setattr(api, "RPUModelForConditionalGeneration",
                            SimpleNamespace(from_pretrained=loader), raising=False)
        monkeypatch.setattr(api, architecture + "Cache", SimpleNamespace(
            from_model=lambda *_a, **_kw: cache), raising=False)
        adapter = ModuleType("rpu_backend.adapters." + adapter_name)
        setattr(adapter, architecture + "Adapter", lambda *_a, **_kw: SimpleNamespace(
            to_rpu=lambda **_kw: model))
        monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
        if not is_moe:
            scope = ModuleType(adapter.__name__ + ".quant_scope")
            scope.validate_legacy_text_profile = lambda _: False
            monkeypatch.setitem(sys.modules, scope.__name__, scope)
            monkeypatch.setattr(runner, "_cpu_fp32_argmax_token",
                                greedy_api.greedy_token_ids)
    registry, transformers = ModuleType("rpu_backend.model_registry"), ModuleType("transformers")
    registry.model_path = lambda _: "/cache/plain-text"
    tokenizer = Mock(return_value=SimpleNamespace(input_ids=ids))
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *_a, **_kw: tokenizer)
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setitem(sys.modules, transformers.__name__, transformers)
    original_to = torch.Tensor.to
    def no_upload(tensor, *args, **kwargs):
        if (args and args[0] == "rpu") or kwargs.get("device") == "rpu":
            pytest.fail("CPU token IDs were uploaded before host embedding")
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", no_upload)
    generate = Mock(wraps=runner._generate)
    monkeypatch.setattr(runner, "_generate", generate)
    target_runner = importlib.import_module("examples.runners." + ("qwen3_5" if target.startswith("qwen3_5_") else target) + ".runner")
    infer, owner = target_runner.prepare({
        "example": {"target": "qwen3_5" if target.startswith("qwen3_5_") else target,
                    "registry_alias": "plain-test", "profile_id": "plain-test"},
        "input": {"prompt": "hello", "decode_steps": 2, "stop_on_eos": False}},
        SimpleNamespace(owner=None))
    result = infer()
    assert owner is model and result["token_ids"].tolist() == [[1, 1, 1]]
    assert generate.call_args.kwargs["reuse_greedy_logits"] is (target in {"qwen3", "llama"})
    assert generate.call_args.kwargs["token_selector"] is (
        greedy_api.greedy_token_ids if target == "qwen3_5_dense" else None)
    assert model.call_args_list[0].kwargs["input_ids"] is ids
    assert all(call.kwargs["input_ids"].device.type == "cpu" for call in model.call_args_list)


def test_dinov3_public_vision_reuses_hidden_readback_with_owned_cls(monkeypatch, greedy_api):
    runner = load("example_dinov3_cpu_cls", "examples/runners/dinov3/runner.py")
    images, pixels, hf_config = [SimpleNamespace(size=(224, 224))], torch.ones((1, 3, 224, 224)), object()
    monkeypatch.setattr(importlib.import_module("examples.common.generation"), "_image_inputs", lambda _: images)
    registry = ModuleType("rpu_backend.model_registry")
    registry.model_path = lambda _: "/cache/dinov3"
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    base = torch.arange(201 * 768).remainder(997).reshape(1, 201, 768).half()
    hidden_values = [base, base + 1, base]
    readbacks = []

    class Readback:
        def __init__(self, value, name):
            self.value, self.name = value, name

        def cpu(self):
            readbacks.append(self.name)
            return self.value.clone()

    # The private adapter result is already CPU-owned. The unused pooler
    # sentinel rejects an accidental second device readback by recording it.
    prepared_hidden = [hidden.clone() for hidden in hidden_values]
    model = Mock(side_effect=[SimpleNamespace(
        last_hidden_state=hidden,
        pooler_output=Readback(hidden[:, 0], "pooler")) for hidden in prepared_hidden])
    model.eval.return_value = model
    loader = Mock(return_value=model)
    processor = Mock(return_value=SimpleNamespace(pixel_values=pixels))
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace()
    transformers.AutoConfig = SimpleNamespace(from_pretrained=lambda *_a, **_kw: hf_config)
    transformers.AutoImageProcessor = SimpleNamespace(from_pretrained=lambda *_a, **_kw: processor)
    transformers.DINOv3ViTModel = SimpleNamespace(from_pretrained=loader)
    monkeypatch.setitem(sys.modules, transformers.__name__, transformers)
    adapter = ModuleType("rpu_backend.adapters.dinov3")
    factory = Mock(return_value=SimpleNamespace(to_rpu=lambda: model))
    factory.preflight = Mock()
    adapter.DINOv3Adapter = factory
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    from examples.common import config as _common
    monkeypatch.setattr(_common, "install_quantization", lambda *_: None)
    execution = {"tensor_parallel_size": 8}
    state = SimpleNamespace(owner=None)
    infer, owner = runner.prepare({
        "example": {"target": "dinov3", "registry_alias": "dino-test", "profile_id": "dino-test"},
        "input": {}, "rpu_execution": execution}, state)
    outputs = [infer() for _ in hidden_values]
    assert owner is state.owner is model
    factory.preflight.assert_called_once_with(hf_config)
    factory.preflight_quantization.assert_not_called()
    factory.assert_called_once_with(model, rpu_execution=execution, quantization=None)
    loader.assert_called_once_with("/cache/dinov3", config=hf_config,
                                  dtype=torch.float16, local_files_only=True)
    assert all(call.args == (pixels,) and
               call.kwargs == {"_rpu_cpu_post_layer_norm": True}
               for call in model.call_args_list)
    assert readbacks == []
    for output, expected, prepared in zip(outputs, hidden_values, prepared_hidden):
        hidden, cls = output["last_hidden_state"], output["pooler_output"]
        assert hidden is prepared and hidden.device.type == "cpu"
        assert torch.equal(hidden, expected) and torch.equal(cls, expected[:, 0])
        assert cls.shape == (1, 768) and cls.dtype == torch.float16
        assert cls.data_ptr() != hidden.data_ptr()
    outputs[1]["pooler_output"].fill_(-1)
    assert torch.equal(outputs[1]["last_hidden_state"], hidden_values[1])
    assert torch.equal(outputs[0]["pooler_output"], outputs[2]["pooler_output"])


@pytest.mark.parametrize("component", ["text", "multimodal", "vision"])
def test_qwen3vl_prepare_reads_current_cpu_ids_and_preserves_cold_geometry(
        monkeypatch, greedy_api, component):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    from examples.common import config as _common
    validate = Mock()
    monkeypatch.setattr(_common, "validate_checkpoint_profile", validate)
    runner = load("example_qwen3vl_cpu_ids", "examples/runners/qwen3_vl/runner.py")
    monkeypatch.setattr(runner, "install_quantization", lambda *_: None)
    ids = torch.tensor([[1, 3 if component != "text" else 2, 0]])
    pixels, grid = torch.ones(4, 6), torch.tensor([[1, 2, 2]])
    mask, types = torch.ones_like(ids), torch.tensor([[0, 1, 0]])
    encoded = {"input_ids": ids, "pixel_values": pixels, "image_grid_thw": grid,
               "attention_mask": mask, "mm_token_type_ids": types}
    tokenizer = Mock(return_value=SimpleNamespace(input_ids=ids))
    tokenizer.decode.return_value = "tokens"
    processor = SimpleNamespace(tokenizer=tokenizer,
                                apply_chat_template=Mock(return_value=encoded))
    registry, transformers = ModuleType("rpu_backend.model_registry"), ModuleType("transformers")
    registry.model_path = lambda _: "/cache/qwen3-vl"
    transformers.AutoTokenizer = SimpleNamespace()
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=lambda *_a, **_kw: processor)
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setitem(sys.modules, transformers.__name__, transformers)
    monkeypatch.setattr(importlib.import_module("examples.common.generation"), "_image_inputs", lambda _: [SimpleNamespace(size=(32, 32))])

    cache, language, visual = Mock(), object(), Mock(return_value="vision-output")
    seen = []
    def forward(**kwargs):
        current = kwargs["input_ids"]
        assert current.device.type == "cpu" and current.dtype == torch.long
        assert current.is_contiguous() and kwargs["past_key_values"] is cache
        if current.shape[1] == 3:
            assert current is ids
            for name in ("pixel_values", "image_grid_thw", "attention_mask", "mm_token_type_ids"):
                if component == "multimodal":
                    assert kwargs[name] is encoded[name]
                else:
                    assert name not in kwargs
        else:
            assert current.shape == (1, 1) and "pixel_values" not in kwargs
        seen.append(current.clone())
        token = current[:, :1].remainder(3)
        return SimpleNamespace(logits=torch.nn.functional.one_hot(token, num_classes=4).float())
    model = Mock(side_effect=forward)
    model.config = SimpleNamespace(architectures=["Qwen3VLForConditionalGeneration"])
    model.generation_config = None
    model.model = SimpleNamespace(language_model=language, visual=visual)
    loader, cache_factory = Mock(return_value=model), Mock(return_value=cache)
    monkeypatch.setattr(greedy_api, "RPUModelForConditionalGeneration",
                        SimpleNamespace(from_pretrained=loader), raising=False)
    monkeypatch.setattr(greedy_api, "RPUCache",
                        SimpleNamespace(from_model=cache_factory), raising=False)
    original_to = torch.Tensor.to
    uploads = []
    def no_ids_upload(tensor, *args, **kwargs):
        if (args and str(args[0]) == "rpu") or str(kwargs.get("device")) == "rpu":
            assert component == "vision" and tensor is pixels
            uploads.append(tensor)
            if args: args = ("cpu", *args[1:])
            if "device" in kwargs: kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", no_ids_upload)
    execution = {"tensor_parallel_size": 8}
    config = {"example": {"target": "qwen3_vl", "registry_alias": "qwen3vl-test", "profile_id": "qwen3vl-test"},
              "input": {"component": component, "prompt": "hello", "decode_steps": 2,
                        "stop_on_eos": False}, "rpu_execution": execution}
    state = SimpleNamespace(owner=None)
    infer, owner = runner.prepare(config, state)
    assert owner is state.owner is model
    validate.assert_called_once_with(config, "/cache/qwen3-vl")
    # Configured image grids size the cold Vision install; text has none.
    grids = {} if component == "text" else {"image_grid_thw": grid}
    loader.assert_called_once_with("/cache/qwen3-vl", dtype=torch.float16, device="rpu",
                                  rpu_execution=execution, local_files_only=True, trust_remote_code=False,
                                  **grids)
    if component == "vision":
        assert infer() == "vision-output" and len(uploads) == 1
        assert visual.call_args.kwargs["grid_thw"] is grid
        assert visual.call_args.args[0].dtype == torch.float16
        cache_factory.assert_not_called()
        model.assert_not_called()
        return
    cache_factory.assert_called_once_with(language, input_ids=ids, max_new_tokens=2)
    alias, version = ids.numpy(), ids._version
    retained = []
    for value in (1, 2, 1):
        alias[0, 0] = value
        result = infer()
        assert ids._version == version and result["token_ids"].tolist() == [[value] * 3]
        assert result["generation"]["prefill_tokens"] == 3
        assert result["generation"]["decode_calls"] == 2
        for old, snapshots in retained:
            for name, snapshot in snapshots.items():
                assert torch.equal(old[name], snapshot)
        retained.append((result, {name: result[name].clone()
                                 for name in ("token_ids", "first_logits", "logits")}))
    assert not uploads and [tuple(value.shape) for value in seen] == [(1, 3), (1, 1), (1, 1)] * 3
    assert torch.equal(seen[0], seen[6]) and not torch.equal(seen[0], seen[3])
    assert cache.reset.call_count == 3


@pytest.mark.parametrize("sequence, generation_eos, config_eos, expected", [
    ([2], None, 2, [2]),
    ([1, 3], [2, 3], None, [1, 3]),
    ([2, 3], 3, 2, [2, 3]),
])
def test_generation_stops_on_checkpoint_eos_without_another_forward(
        sequence, generation_eos, config_eos, expected, greedy_api):
    runner = load("example_eos_test", "examples/common/generation.py")
    cache = Mock()
    outputs = [SimpleNamespace(logits=torch.nn.functional.one_hot(
        torch.tensor([[token]]), num_classes=4).float()) for token in sequence]
    model = Mock(side_effect=outputs)
    model.config = SimpleNamespace(eos_token_id=config_eos)
    model.generation_config = SimpleNamespace(eos_token_id=generation_eos)
    result = runner._generate(model, cache, torch.tensor([[0, 1]]), 31)
    assert result["token_ids"].tolist() == [expected]
    assert model.call_count == len(expected)
    assert greedy_api.greedy_token_ids.call_count == len(expected)
    assert torch.equal(result["logits"], outputs[-1].logits)
    cache.reset.assert_called_once_with()


def test_generation_eos_falls_back_to_nested_text_config_then_tokenizer():
    runner = load("example_nested_eos_test", "examples/common/generation.py")
    model = SimpleNamespace(config=SimpleNamespace(
        text_config=SimpleNamespace(eos_token_id=[2, 3])))
    assert runner._eos_token_ids(model, SimpleNamespace(eos_token_id=1)) == {2, 3}
    model.config.text_config.eos_token_id = None
    assert runner._eos_token_ids(model, SimpleNamespace(eos_token_id=1)) == {1}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("via_public_llama", [False, True])
def test_fixed_decode_ignores_eos_and_keeps_owned_tokens_and_phase_timings(
        monkeypatch, greedy_api, dtype, via_public_llama):
    runner = load("example_fixed_decode_test", "examples/common/generation.py")
    # A reusable output buffer models the backend's shared logits storage.
    buffer = torch.zeros(1, 1, 4, dtype=dtype)
    sequence = iter([2, 1, 3])

    def forward(**_kwargs):
        buffer.zero_()
        buffer[0, 0, next(sequence)] = 5
        return SimpleNamespace(logits=buffer)

    model = Mock(side_effect=forward)
    model.config = SimpleNamespace(eos_token_id=2)
    model.generation_config = None
    ticks = iter([0., .5, 1., 1.1, 2., 2.2])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))
    input_ids, cache = torch.tensor([[0, 1, 0]]), Mock()
    if via_public_llama:
        infer = _prepare_public_llama_infer(
            monkeypatch, runner, model, cache, input_ids, 2, False)
        out = infer()
    else:
        out = runner._generate(model, cache, input_ids, 2,
                               stop_on_eos=False, reuse_greedy_logits=True)
    assert out["token_ids"].tolist() == [[2, 1, 3]]
    assert out["first_logits"].argmax().item() == 2
    assert out["first_logits"].dtype == torch.float32
    assert out["logits"].dtype == dtype
    assert model.call_count == 3
    stats = out["generation"]
    assert stats["prefill_tokens"] == 3 and stats["decode_calls"] == 2
    assert stats["prefill_ms"] == 500.
    assert stats["decode_ms"] == pytest.approx(300.)
    assert stats["decode_step_ms"] == pytest.approx([100., 200.])
    assert stats["prefill_tokens_per_second"] == 6.
    assert stats["decode_tokens_per_second"] == pytest.approx(2 / .3)


@pytest.mark.parametrize("decode_steps,stop_on_eos", [(0, False), (2, False), (2, True)])
@pytest.mark.parametrize("rpu_origin", [False, True])
@pytest.mark.parametrize("via_public_llama", [False, True])
def test_plain_greedy_reuses_readback_with_ties_and_owned_snapshots(
        monkeypatch, greedy_api, decode_steps, stop_on_eos, rpu_origin, request, via_public_llama):
    runner = load("example_greedy_readback_test", "examples/common/generation.py")
    clock, reads, native_calls = [0.], [], []
    buffer = torch.zeros((1, 1, 4), dtype=torch.float16)
    class RPULogits(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="rpu")
    model_logits = buffer.as_subclass(RPULogits) if rpu_origin else buffer
    native_reduce = request.getfixturevalue("native_cpu_argmax") if rpu_origin else None
    values = [torch.tensor([4., 4., 1., 0.], dtype=torch.float16),
              torch.tensor([-0., 0., -1., -1.], dtype=torch.float16),
              torch.tensor([1., float("nan"), 3., float("nan")], dtype=torch.float16)]
    pending = iter(values)

    def forward(**_kwargs):
        clock[0] += .1
        buffer.copy_(next(pending))
        return SimpleNamespace(logits=model_logits)

    original_cpu = torch.Tensor.cpu
    def readback(tensor, *args, **kwargs):
        # Strip only the fake device wrapper before PyTorch redispatches the
        # patched method; one simulated device read yields a normal CPU Tensor.
        value = tensor.as_subclass(torch.Tensor) if isinstance(tensor, RPULogits) else tensor
        value = original_cpu(value, *args, **kwargs)
        if tensor.is_floating_point():
            reads.append((tuple(tensor.shape), tensor.dtype))
            clock[0] += .02
            # Model a CPU readback without borrowing mutable device storage.
            return value.clone()
        return value

    select = greedy_api.greedy_token_ids
    def greedy(logits):
        assert logits.device.type == "cpu" and logits.dtype == torch.float16
        clock[0] += .03
        return select(logits)

    def native(logits):
        assert rpu_origin and logits.device.type == "cpu"
        assert logits.dtype == torch.float16 and logits.is_contiguous()
        native_calls.append(logits.data_ptr())
        clock[0] += .03
        return native_reduce(logits)

    monkeypatch.setattr(torch.Tensor, "cpu", readback)
    monkeypatch.setattr(greedy_api, "greedy_token_ids", greedy)
    monkeypatch.setattr(torch.ops.rpu, "argmax_lastdim_host", native, raising=False)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    model = Mock(side_effect=forward)
    model.config = SimpleNamespace(eos_token_id=0)
    model.generation_config = None
    input_ids, cache = torch.tensor([[1, 2]]), Mock()
    if via_public_llama:
        infer = _prepare_public_llama_infer(
            monkeypatch, runner, model, cache, input_ids, decode_steps, stop_on_eos)
        result = infer()
    else:
        result = runner._generate(model, cache, input_ids, decode_steps,
            stop_on_eos=stop_on_eos, reuse_greedy_logits=True)
    expected_tokens = [0] if stop_on_eos or decode_steps == 0 else [0, 0, 1]
    assert result["token_ids"].tolist() == [expected_tokens]
    assert result["token_ids"].dtype == torch.int64
    assert len(reads) == model.call_count == len(expected_tokens)
    assert len(native_calls) == (len(expected_tokens) if rpu_origin else 0)
    assert select.call_count == (0 if rpu_origin else len(expected_tokens))
    assert all(shape == (1, 1, 4) and dtype == torch.float16 for shape, dtype in reads)
    assert result["first_logits"].dtype == torch.float32
    assert torch.equal(result["first_logits"], values[0].unsqueeze(0).float())
    assert result["logits"].dtype == torch.float16
    expected_final = values[len(expected_tokens) - 1].reshape(1, 1, 4)
    torch.testing.assert_close(result["logits"], expected_final, rtol=0, atol=0, equal_nan=True)
    assert result["generation"]["prefill_ms"] == pytest.approx(150.)
    assert result["generation"]["decode_ms"] == pytest.approx(150. * (len(expected_tokens) - 1))
    buffer.fill_(99.)
    assert torch.equal(result["first_logits"], values[0].unsqueeze(0).float())
    torch.testing.assert_close(result["logits"], expected_final, rtol=0, atol=0, equal_nan=True)


def test_dense_fp32_selector_preserves_ulp_logits_and_stays_in_phase_timers(monkeypatch):
    runner = load("example_fp32_argmax_test", "examples/common/generation.py")
    clock = [0.]
    high = torch.nextafter(torch.tensor(1.), torch.tensor(float("inf")))
    outputs = [torch.tensor([[[1., high, 0.]]]), torch.tensor([[[high, 1., 0.]]])]
    pending = iter(outputs)
    seen = []

    def forward(**_kwargs):
        clock[0] += .1
        return SimpleNamespace(logits=next(pending))

    def native_argmax(logits):
        assert logits.dtype == torch.float32 and logits.device.type == "cpu"
        assert logits.is_contiguous()
        seen.append(logits.data_ptr())
        clock[0] += .02
        return logits.argmax(dim=-1)

    monkeypatch.setattr(torch.ops.rpu, "argmax_lastdim_host", native_argmax, raising=False)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    result = runner._generate(forward, Mock(), torch.tensor([[0, 1]]), 1,
                              stop_on_eos=False, token_selector=runner._cpu_fp32_argmax_token,
                              reuse_greedy_logits=True)
    assert result["token_ids"].tolist() == [[1, 0]]
    assert seen == [value.data_ptr() for value in outputs]
    assert result["first_logits"].dtype == result["logits"].dtype == torch.float32
    assert torch.equal(result["first_logits"], outputs[0][:, -1])
    assert torch.equal(result["logits"], outputs[1])
    assert result["generation"]["prefill_ms"] == pytest.approx(120.)
    assert result["generation"]["decode_ms"] == pytest.approx(120.)


def test_dense_fp32_selector_rejects_precision_changes_before_native_call(monkeypatch):
    runner = load("example_fp32_argmax_dtype_test", "examples/common/generation.py")
    native = Mock()
    monkeypatch.setattr(torch.ops.rpu, "argmax_lastdim_host", native, raising=False)
    with pytest.raises(ValueError, match="CPU FP32"):
        runner._cpu_fp32_argmax_token(torch.zeros((1, 4), dtype=torch.float16))
    native.assert_not_called()


def test_report_input_eos_fill_and_multimodal_extension_keep_image_and_chat_tokens():
    runner = load("example_report_input_test", "examples/common/generation.py")
    tokenizer = Mock(return_value=SimpleNamespace(input_ids=torch.tensor([[10, 11]])))
    ids = runner._text_input_ids(tokenizer, {
        "prompt_token_ids": [1, 2], "prefill_tokens": 5, "prefill_fill_token_id": 99})
    assert ids.tolist() == [[1, 2, 99, 99, 99]]
    tokenizer.assert_not_called()
    # 90=image, 91=user im_end, 92=assistant generation prefix.
    tokenizer.convert_tokens_to_ids.return_value = 91
    encoded = {"input_ids": torch.tensor([[80, 90, 12, 91, 92]]),
               "mm_token_type_ids": torch.tensor([[0, 1, 0, 0, 0]]),
               "attention_mask": torch.ones((1, 5), dtype=torch.long)}
    result = runner._extend_multimodal_input(encoded, {"prompt": "text", "prefill_tokens": 8}, tokenizer)
    assert result["input_ids"].tolist() == [[80, 90, 12, 10, 11, 10, 91, 92]]
    assert result["mm_token_type_ids"].tolist() == [[0, 1, 0, 0, 0, 0, 0, 0]]
    assert result["attention_mask"].tolist() == [[1] * 8]
    with pytest.raises(ValueError, match="cannot truncate"):
        runner._extend_multimodal_input(result, {"prefill_tokens": 2}, tokenizer)


@pytest.mark.parametrize("mutation", [None, "short", "prompt_hash", "ids_hash", "eos", "pad"])
def test_moe_frozen_prose_truncates_without_filling_and_rejects_changed_inputs(tmp_path, mutation):
    runner = load("example_prose_input_test", "examples/common/generation.py")
    path = tmp_path / "prose.txt"
    path.write_text("A complete paragraph from the frozen benchmark input.")
    ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    inputs = {"prompt_file": str(path), "prefill_tokens": 4,
              "prompt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "input_ids_sha256": hashlib.sha256(ids[:, :4].numpy().tobytes()).hexdigest()}
    if mutation == "short":
        ids = ids[:, :3]
    elif mutation == "prompt_hash":
        path.write_text("Changed prose")
    elif mutation == "ids_hash":
        ids[0, 0] = 42
    tokenizer = Mock(return_value={"input_ids": ids})
    tokenizer.eos_token_id = 10 if mutation == "eos" else 90
    tokenizer.pad_token_id = 10 if mutation == "pad" else 91
    if mutation is not None:
        with pytest.raises(ValueError):
            runner._realtext_input_ids(tokenizer, inputs)
        return
    actual, metadata = runner._realtext_input_ids(tokenizer, inputs)
    assert actual.tolist() == [[10, 11, 12, 13]]
    assert metadata["source_tokens"] == 6 and metadata["padding_tokens"] == 0
    assert metadata["input_ids_sha256"] == inputs["input_ids_sha256"]
    tokenizer.assert_called_once_with(path.read_text(), add_special_tokens=False, return_tensors="pt")


def test_moe_forward_argmax_keeps_snapshots_outside_timing_and_records_actual_plan(monkeypatch, greedy_api):
    from types import MappingProxyType

    runner = load("example_moe_timing_test", "examples/common/generation.py")
    clock = [0.]
    inner = SimpleNamespace()
    cache = SimpleNamespace(max_seq_len=7, allocated_max_seq_len=128, position=0)
    cache.reset = lambda: setattr(cache, "position", 0)
    outputs = iter([2, 1, 3, 0])  # EOS first still requires all three decode calls.
    buffer = torch.zeros((1, 1, 4))
    seen = []

    def forward(**kwargs):
        clock[0] += .1
        ids = kwargs["input_ids"]
        seen.append(ids.tolist())
        assert kwargs["logits_to_keep"] == 1
        cache.position += ids.shape[1]
        inner._rpu_last_execution_plan = MappingProxyType({
            "stage": "prefill" if ids.shape[1] > 1 else "decode",
            "chunk_size": 64 if ids.shape[1] > 1 else 1,
            "physical_descriptor": (4, 8), "detail": MappingProxyType({"selected": True})})
        buffer.zero_()
        buffer[0, 0, next(outputs)] = 5
        return SimpleNamespace(logits=buffer)

    model = Mock(side_effect=forward)
    model.model = SimpleNamespace(language_model=inner)
    model.config = SimpleNamespace(eos_token_id=2)
    original_cpu = torch.Tensor.cpu

    def costly_snapshot(tensor, *args, **kwargs):
        if tensor.is_floating_point():
            clock[0] += .9
        return original_cpu(tensor, *args, **kwargs)

    def argmax(logits):
        clock[0] += .02
        return logits.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr(torch.Tensor, "cpu", costly_snapshot)
    monkeypatch.setattr(greedy_api, "greedy_token_ids", argmax)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    result = runner._generate(model, cache, torch.tensor([[10, 11, 12, 13]]), 3,
                              stop_on_eos=False, timing_mode="forward_argmax",
                              input_metadata={"prompt_sha256": "frozen"},
                              reuse_greedy_logits=True)
    assert seen == [[[10, 11, 12, 13]], [[2]], [[1]], [[3]]]
    assert result["token_ids"].tolist() == [[2, 1, 3, 0]]
    assert result["first_logits"].argmax().item() == 2
    stats = result["generation"]
    assert stats["prefill_ms"] == pytest.approx(120.)
    assert stats["decode_ms"] == pytest.approx(360.)
    assert stats["input"]["prompt_sha256"] == "frozen"
    assert stats["execution_plans"]["prefill"]["chunk_size"] == 64
    assert stats["execution_plans"]["decode"]["chunk_size"] == 1
    assert stats["cache"] == {"logical_max_seq_len": 7, "allocated_max_seq_len": 128, "used_positions": 7}


def test_p4096_d4096_means_4096_decode_forwards_and_8192_cache_positions(greedy_api):
    runner = load("example_long_decode_test", "examples/common/generation.py")
    position = 0

    def reset():
        nonlocal position
        position = 0

    def forward(**kwargs):
        nonlocal position
        position += kwargs["input_ids"].shape[1]
        assert position <= 8192
        return SimpleNamespace(logits=torch.tensor([[[1., 2.]]]))

    result = runner._generate(forward, SimpleNamespace(reset=reset),
                              torch.zeros((1, 4096), dtype=torch.long), 4096,
                              stop_on_eos=False)
    assert position == 8192
    assert result["token_ids"].shape == (1, 4097)
    assert result["generation"]["decode_calls"] == 4096


def test_shared_hwperf_preserves_kernel_aggregation(tmp_path):
    performance = load("performance_test", "python/rpu_backend/runtime/performance.py")
    trace = tmp_path / "rpu_hwperf_test.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"ph": "B", "ts": 10},
        {"ph": "E", "ts": 30, "cat": "Compute", "name": "linear_1", "args": {"duration_us": 20}},
        {"ph": "E", "ts": 60, "cat": "Compute", "name": "linear_2", "args": {"duration_us": 30}},
        {"ph": "E", "ts": 70, "cat": "DMA", "name": "copy_3", "args": {"duration_us": 10}},
    ]}))
    expected = performance._hwperf_aggregate_file(str(trace))
    assert expected["latency_ms"] == 0.06
    assert expected["rows"] == [{"name": "linear", "count": 2, "total_us": 50.,
                                 "avg_us": 25., "pct": 100.}]
    assert "linear" in performance._hwperf_format_block(expected)
    assert performance._hwperf_collect_files(str(tmp_path)) == [str(trace)]


@pytest.mark.parametrize("fail, support_status, evaluation_only", [
    (None, "SUPPORTED", False),
    (None, "CONTROLLED_EVALUATION", True),
    (None, "IMPLEMENTED_UNVERIFIED_HW", True),
    (None, "EXPERIMENTAL", True),
    ("inference", "SUPPORTED", False),
    ("cleanup", "SUPPORTED", False),
    ("inference_cleanup", "SUPPORTED", False),
    ("inference_report", "SUPPORTED", False),
    ("report", "SUPPORTED", False),
    ("hwperf", "SUPPORTED", False),
])
def test_example_times_actual_inference_and_records_failures(
        monkeypatch, tmp_path, fail, support_status, evaluation_only):
    runner = load("public_runner_test", "examples/common/measurement.py")
    performance = load("performance_runner_test", "python/rpu_backend/runtime/performance.py")
    common = importlib.import_module("examples.common.config")
    monkeypatch.setattr(common, "cpu_runtime", lambda: load("example_cpu_test", "examples/common/cpu.py"))
    monkeypatch.setattr(common, "example_profiles", lambda: {
        "test.profile": {"id": "test.profile", "support": {"status": support_status}}})
    monkeypatch.setitem(sys.modules, "rpu_backend", ModuleType("rpu_backend"))
    monkeypatch.setitem(sys.modules, "rpu_backend.runtime.performance", performance)
    # Match the live backend: device calls are synchronous and this namespace
    # intentionally has no CUDA-style synchronize method.
    monkeypatch.setattr(torch, "rpu", SimpleNamespace(
        hw_perf_trace=lambda *_a, **_kw: contextlib.nullcontext(),
        _is_in_bad_fork=lambda: False, manual_seed_all=lambda seed: None), raising=False)
    output = torch.tensor([1., 2.])
    samples = [{"logits": output, "generation": {
        "prefill_tokens": 64, "decode_calls": 2,
        "prefill_ms": ms, "decode_ms": ms,
        "prefill_tokens_per_second": 64000 / ms,
        "decode_tokens_per_second": 2000 / ms,
    }} for ms in (10., 100., 300.)]
    infer = Mock(side_effect=samples)
    owner = SimpleNamespace(close=Mock())
    primary = RuntimeError("forward failed")
    if fail in {"inference", "inference_cleanup", "inference_report"}:
        infer.side_effect = primary
    if fail in {"cleanup", "inference_cleanup"}:
        owner.close.side_effect = RuntimeError("close failed")
    if fail in {"report", "inference_report"}:
        original_write = Path.write_text
        def write(path, *args, **kwargs):
            if path.name == "report.json":
                raise OSError("report disk full")
            return original_write(path, *args, **kwargs)
        monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(runner, "_prepare", lambda config: (infer, owner))
    config = {"example": {"profile_id": "test.profile", "target": "qwen3"}, "input": {},
              "run": {"output_dir": str(tmp_path), "warmup": 1, "warmup_decode_steps": 2,
                      "runs": 2, "hwperf": fail == "hwperf", "inference_mode": False}}
    if fail:
        with pytest.raises(OSError if fail == "report" else RuntimeError) as raised:
            runner.run(config)
        if fail in {"report", "inference_report"}:
            owner.close.assert_called_once()
            assert not list(tmp_path.glob("*/report.json"))
            if fail == "inference_report":
                assert raised.value is primary
                if hasattr(primary, "add_note"):
                    assert any("report disk full" in note for note in primary.__notes__)
            else:
                assert str(raised.value) == "report disk full"
            return
    else:
        assert runner.run(config) == 0
        assert infer.call_count == 3
        assert infer.call_args_list[0].kwargs == {"decode_steps": 2}
        assert all(not call.kwargs for call in infer.call_args_list[1:])
    report_path, = tmp_path.glob("*/report.json")
    report = json.loads(report_path.read_text())
    assert report["status"] == ("FAIL" if fail else "PASS")
    assert report["execution"]["status"] == ("FAILED" if fail else "COMPLETED")
    assert report["validation"]["reference"]["status"] == "NOT_RUN"
    assert report["validation"]["task"]["status"] == "NOT_EVALUATED"
    assert report["artifacts"]["report"] == str(report_path)
    if report["wall_ms_runs"]:
        assert report["timing"][0]["samples"] == report["wall_ms_runs"]
    assert report["correctness"] == "not_run"
    assert report["formal_certification"] is False
    assert report["support"]["status"] == support_status
    assert report["evaluation_only"] is evaluation_only
    assert "evaluation_only" not in report["config"]["example"]
    assert report["cpu_runtime"]["inference_mode"] is False
    owner.close.assert_called_once()
    assert report_path.with_name("report.md").exists() is (fail is None)
    if fail == "inference_cleanup":
        assert report["error"] == "RuntimeError: forward failed"
        assert report["cleanup_error"] == "RuntimeError: close failed"
    if not fail:
        assert str(report_path.with_name("report.md")) in report["artifacts"]["outputs"]
        assert len(report["wall_ms_runs"]) == 2
        assert torch.equal(torch.load(report_path.with_name("output.pt"), weights_only=True)["logits"], output)
        assert len(report["generation_runs"]) == 2
        # Match the report's throughput from mean latency, not mean token/s.
        assert report["generation_summary"]["prefill_tokens_per_second"] == 320.
        assert report["generation_summary"]["decode_tokens_per_second"] == 10.
        prefill_rate, = [row for row in report["timing"] if row["metric"] == "prefill_throughput"]
        assert prefill_rate["unit"] == "tokens/s"
        assert prefill_rate["statistic"] == "aggregate_tokens_over_total_time"
        assert prefill_rate["value"] == 320.


def test_cleanup_failure_never_prints_success(monkeypatch, tmp_path, capsys):
    test_example_times_actual_inference_and_records_failures(
        monkeypatch, tmp_path, "cleanup", "SUPPORTED", False)
    assert "inference PASS:" not in capsys.readouterr().out


def test_stage_timings_time_declared_calls_once_and_restore_owners(monkeypatch):
    results = importlib.import_module("examples.common.results")
    clock = iter(range(100))
    monkeypatch.setattr(results.time, "perf_counter", lambda: next(clock))

    class Runtime:
        def denoise(self, steps):
            # A stage method reached again inside the same stage is one span.
            return self.denoise(steps - 1) if steps else "action"

    runtime = Runtime()
    local = runtime.vision = lambda: "tokens"
    boundaries = (("vision", runtime, "vision"), ("denoise", runtime, "denoise"),
                  ("vision", Runtime, "denoise"))
    with pytest.raises(LookupError):
        with results.stage_timings(boundaries[:2]) as phases:
            assert runtime.vision() == "tokens" and runtime.denoise(2) == "action"
            assert runtime.vision() == "tokens"
            assert phases == {"vision_ms": 2000., "denoise_ms": 1000.}
            raise LookupError("inference failed")
    assert runtime.vision is local and "denoise" not in vars(runtime)
    with pytest.raises(TypeError, match="vision: Runtime.missing is missing or not callable"):
        with results.stage_timings(boundaries[:2] + (("vision", runtime, "missing"),)):
            pass
    runtime.missing = None
    with pytest.raises(TypeError, match="Runtime.missing is missing or not callable"):
        with results.stage_timings(boundaries[:2] + (("vision", runtime, "missing"),)):
            pass
    assert runtime.vision is local and "denoise" not in vars(runtime)
    with results.stage_timings(boundaries[2:]) as phases:
        assert runtime.denoise(0) == "action" and phases == {"vision_ms": 1000.}
    assert vars(Runtime)["denoise"].__name__ == "denoise"


def test_stage_phase_timings_keep_only_stages_measured_in_every_run():
    results = importlib.import_module("examples.common.results")
    rows = [{"vision_ms": 1., "denoise_ms": 4.}, {"vision_ms": 3., "denoise_ms": 6., "prefill_ms": 2.}]
    timings = results.stage_phase_timings(rows, "stage calls")
    assert [(item["metric"], item["value"], item["samples"]) for item in timings] == [
        ("vision_latency", 2., [1., 3.]), ("denoise_latency", 5., [4., 6.])]
    assert results.stage_phase_timings([{}, {}], "stage calls") == []
    assert results.stage_phase_timings([], "stage calls") == []


def test_measurement_times_declared_stages_in_measured_runs_only(monkeypatch, tmp_path):
    runner = load("public_runner_stage_test", "examples/common/measurement.py")
    performance = load("performance_runner_stage_test", "python/rpu_backend/runtime/performance.py")
    common = importlib.import_module("examples.common.config")
    monkeypatch.setattr(common, "cpu_runtime", lambda: load("example_cpu_stage_test", "examples/common/cpu.py"))
    monkeypatch.setattr(common, "example_profiles", lambda: {"test.profile": {"id": "test.profile"}})
    monkeypatch.setitem(sys.modules, "rpu_backend", ModuleType("rpu_backend"))
    monkeypatch.setitem(sys.modules, "rpu_backend.runtime.performance", performance)
    monkeypatch.setattr(runner, "_provenance", lambda config: {})

    class Policy:
        calls = 0

        def vision(self):
            return torch.ones(2)

        def denoise(self, features):
            return features * 2

        def predict(self):
            self.calls += 1
            return self.denoise(self.vision())

        def close(self):
            pass

    policy = Policy()

    def infer():
        return policy.predict()
    # An owner without a declared method leaves that stage unmeasured; the run still passes.
    infer.stage_boundaries = (("vision", policy, "vision"), ("prefill", policy, "prefill"),
                              ("denoise", policy, "denoise"), ("action", None, "forward"))
    monkeypatch.setattr(runner, "_prepare", lambda config: (infer, policy))
    runner.run({"example": {"profile_id": "test.profile", "target": "hy_vla"}, "input": {},
                "run": {"output_dir": str(tmp_path), "warmup": 2, "runs": 3}})
    report = json.loads(next(tmp_path.glob("*/report.json")).read_text())
    assert policy.calls == 5 and "vision" not in vars(policy)
    assert report["status"] == "PASS"
    assert report["stage_boundaries"] == ["vision: Policy.vision", "denoise: Policy.denoise"]
    assert report["stage_boundaries_missing"] == ["prefill: prefill", "action: forward"]
    assert len(report["stage_runs"]) == 3
    assert all(set(row) == {"vision_ms", "denoise_ms"} for row in report["stage_runs"])
    metrics = {item["metric"]: item for item in report["timing"]}
    assert list(metrics) == ["total_latency", "vision_latency", "denoise_latency"]
    assert metrics["vision_latency"]["samples"] == [row["vision_ms"] for row in report["stage_runs"]]
    assert all(row["vision_ms"] + row["denoise_ms"] <= total
               for row, total in zip(report["stage_runs"], report["wall_ms_runs"]))


def test_public_examples_do_not_import_certification_harnesses():
    import ast

    for path in (ROOT / "examples").glob("*.py"):
        source = path.read_text()
        assert "test_v5_e2e_perf" not in source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert ".tests" not in (node.module or "")
            elif isinstance(node, ast.Import):
                assert all(".tests" not in alias.name for alias in node.names)


def test_returned_output_has_independent_storage_and_rejects_nonfinite():
    runner = load("public_runner_output_test", "examples/common/measurement.py")
    original = torch.tensor([1.])
    copied = runner._cpu_output({"action": original})
    original.fill_(2.)
    assert copied["action"].item() == 1.
    with pytest.raises(ValueError, match="NaN/Inf"):
        runner._cpu_output(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="NaN/Inf"):
        runner._cpu_output({"score": float("inf")})


def test_provenance_binds_config_tracked_untracked_and_dependency_bytes(monkeypatch, tmp_path):
    runner = load("example_provenance_test", "examples/common/measurement.py")
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "examples/common/measurement.py"))
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    source = tmp_path / "source.py"
    source.write_text("original\n")
    git("add", "source.py")
    git("-c", "user.name=RhinoForge", "-c", "user.email=",
        "commit", "-qm", "fixture")
    clean = runner._provenance({"run": {"runs": 1}})
    assert clean["source"]["dirty"] is False
    assert clean["dependencies"] == {"status": "UNAVAILABLE"}
    source.write_text("changed\n")
    dirty = runner._provenance({"run": {"runs": 1}})
    assert dirty["source"]["dirty"] is True
    assert dirty["source"]["worktree_delta_sha256"] != clean["source"]["worktree_delta_sha256"]
    (tmp_path / "new.py").write_text("new source\n")
    untracked = runner._provenance({"run": {"runs": 1}})
    assert untracked["source"]["worktree_delta_sha256"] != dirty["source"]["worktree_delta_sha256"]
    manifest = {"schema_version": 1, "launch_kernel": {"commit": "launch"},
                "artifacts": {"rhino_ops_main": {"commit": "main"},
                              "rhino_ops_expansion": {"commit": "expansion"}}}
    raw = json.dumps(manifest).encode()
    (tmp_path / "release").mkdir(parents=True, exist_ok=True)
    (tmp_path / "release/runtime-dependencies.json").write_bytes(raw)
    assert not (tmp_path / "manifests").exists()
    changed = runner._provenance({"run": {"runs": 2}})
    assert changed["config_sha256"] != clean["config_sha256"]
    assert changed["source"]["commit"] == clean["source"]["commit"]
    assert changed["dependencies"]["manifest_sha256"] == hashlib.sha256(raw).hexdigest()
    assert changed["dependencies"]["scope"] == "declared_build_inputs"
    assert changed["dependencies"]["launch_kernel"] == "launch"
    assert changed["dependencies"]["rhino_ops"] == {"main": "main", "expansion": "expansion"}


def test_hw_profile_context_is_enabled_before_prepare(monkeypatch, tmp_path):
    runner = load("example_hw_profile_test", "examples/common/measurement.py")
    performance = load("example_hw_profile_performance", "python/rpu_backend/runtime/performance.py")
    common = importlib.import_module("examples.common.config")
    monkeypatch.setattr(common, "cpu_runtime", lambda: load("example_cpu_test", "examples/common/cpu.py"))
    monkeypatch.setattr(common, "example_profiles", lambda: {"test": {"id": "test", "support": {"status": "SUPPORTED"}}})
    for name, value in (("rpu_backend", ModuleType("rpu_backend")),
                        ("rpu_backend.runtime.performance", performance)):
        monkeypatch.setitem(sys.modules, name, value)
    events = []
    @contextlib.contextmanager
    def trace(directory, *, max_dumps):
        assert max_dumps == 7
        events.append("trace_on")
        Path(directory).mkdir()
        (Path(directory) / "rpu_hwperf_test.json").write_text(json.dumps({"traceEvents": [
            {"ph": "B", "ts": 1},
            {"ph": "E", "ts": 11, "cat": "Compute", "name": "linear_1",
             "args": {"duration_us": 10}},
        ]}))
        yield
        events.append("trace_off")
    monkeypatch.setattr(torch, "rpu", SimpleNamespace(
        hw_perf_trace=trace,
        _is_in_bad_fork=lambda: False, manual_seed_all=lambda seed: None), raising=False)
    def prepare(config):
        assert events == ["trace_on"]
        events.append("prepare_build")
        return lambda: torch.tensor([1.]), SimpleNamespace(close=lambda: events.append("close"))
    monkeypatch.setattr(runner, "_prepare", prepare)
    config = {"example": {"profile_id": "test", "target": "qwen3"}, "input": {},
              "run": {"warmup": 0, "runs": 1, "output_dir": str(tmp_path)},
              "runner": {"hw_profile": {"enabled": True, "max_dumps": 7}}}
    assert runner.run(config) == 0
    assert events == ["trace_on", "prepare_build", "trace_off", "close"]
    path, = tmp_path.glob("*/report.json")
    report = json.loads(path.read_text())
    assert report["hwperf_max_dumps"] == 7
    assert report["hwperf"][0]["rows"][0]["name"] == "linear"
