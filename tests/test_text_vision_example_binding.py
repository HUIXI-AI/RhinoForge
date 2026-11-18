"""Qwen3.5 example must bind its cold request on the first public load."""

import importlib.util
from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def example_entry_path(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))


def load(monkeypatch, name, path):
    if path.startswith("examples/"):
        return import_module(path.removesuffix(".py").replace("/", "."))
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("chunk", ["auto", 640])
@pytest.mark.parametrize("legacy", [False, True])
def test_qwen35_first_loader_and_adapter_use_the_same_real_cold_binding(monkeypatch, chunk, legacy):
    from examples.common import generation
    from rpu_backend.adapters.qwen3_5 import quant_scope

    if legacy and chunk != "auto":
        chunk = 128
    execution = load(monkeypatch, "example_real_execution", "python/rpu_backend/api/_execution.py")
    runner = load(monkeypatch, "example_text_binding", "examples/common/preparation.py")
    generate = Mock()
    monkeypatch.setattr(generation, "_generate", generate)
    request = {"prefill": {"chunk_size": chunk}}
    text_config = SimpleNamespace()
    model = SimpleNamespace(config=text_config if legacy else SimpleNamespace(text_config=text_config))
    monkeypatch.setattr(quant_scope, "validate_legacy_text_profile", lambda _: legacy)
    allocator = Mock()
    ids = Mock(shape=(1, 17))
    tokenizer = Mock(return_value=SimpleNamespace(input_ids=ids))
    cache = object()
    install = Mock(return_value=model)

    def public_load(*_args, **kwargs):
        # This is the production binder the public loader invokes via its
        # first Qwen3_5Adapter; no permissive constructor mock can hide drift.
        execution.bind_rpu_execution(model, kwargs.get("rpu_execution"),
                                     entry_point="public loader")
        return model

    class Adapter:
        def __init__(self, raw, *, rpu_execution):
            execution.bind_rpu_execution(raw, rpu_execution, entry_point="adapter")

        def to_rpu(self, **kwargs):
            return install(**kwargs)

    for name, attributes in {
        "torch": {"float16": "fp16", "rpu": SimpleNamespace(set_caching_allocator=allocator)},
        "transformers": {"AutoTokenizer": SimpleNamespace(from_pretrained=lambda *_a, **_k: tokenizer)},
        "rpu_backend.model_registry": {"model_path": lambda _: "/cache/qwen35"},
        "rpu_backend.adapters.qwen3_5": {"Qwen3_5Adapter": Adapter},
        "rpu_backend.api": {
            "RPUModelForConditionalGeneration": SimpleNamespace(from_pretrained=public_load),
            "Qwen3_5Cache": SimpleNamespace(from_model=lambda *_a, **_k: cache),
        },
    }.items():
        stub = ModuleType(name)
        vars(stub).update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)

    infer, owner = runner.prepare({
        "example": {"target": "qwen3_5", "registry_alias": "qwen3_5-2b",
                    "profile_id": "qwen3_5-2b.fp16.text"},
        "input": {"prompt": "Hello", "decode_steps": 4}, "rpu_execution": request,
    })
    assert owner is model and callable(infer)
    infer()
    assert generate.call_args.kwargs["token_selector"] is (
        None if legacy else generation._cpu_fp32_argmax_token)
    assert model._rpu_execution == request
    install.assert_called_once_with(max_seq_len=21)
    if legacy:
        allocator.assert_called_once_with(False)
    else:
        allocator.assert_not_called()


@pytest.mark.parametrize("cores", [4, 6, 8])
@pytest.mark.parametrize("failure", [None, "install", "cache"])
@pytest.mark.parametrize("prefill_length,decode_steps", [(17, 31), (4096, 4096)])
def test_qwen35_moe_text_example_uses_exact_public_core_profile(
        monkeypatch, cores, failure, prefill_length, decode_steps):
    from examples.common import generation
    common = load(monkeypatch, "example_moe_text_common", "examples/common/config.py")
    runner = load(monkeypatch, "example_moe_text_binding", "examples/common/preparation.py")
    generate = Mock()
    monkeypatch.setattr(generation, "_generate", generate)
    execution = load(monkeypatch, "example_moe_text_execution", "python/rpu_backend/api/_execution.py")
    config = common.load_config(ROOT / "examples/configs/qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn.toml")
    config["rpu_execution"]["model"]["num_cores"] = cores
    common._validate_execution(config["rpu_execution"],
        profile=common.planner_profiles()[config["example"]["profile_id"]])
    config["input"]["decode_steps"] = decode_steps
    request = config["rpu_execution"]
    model = SimpleNamespace(config=SimpleNamespace(
        architectures=["Qwen3_5MoeForConditionalGeneration"]), close=Mock())
    # Mirrors Qwen3_5MoeAdapter._EXECUTION_SUPPORTED: the adapter needs torch,
    # so this host test cannot import the declaration it stands in for.
    supported = {"model": ("num_cores",),
                 "prefill": ("chunk_size", "padding_rows", "padding_budget",
                             "linear_acc32", "moe_resident_prefill")}
    ids = Mock(shape=(1, prefill_length))
    tokenizer = Mock(return_value=SimpleNamespace(input_ids=ids))
    install = Mock(return_value=model)
    cache_factory = Mock(return_value=object())
    if failure == "install":
        install.side_effect = RuntimeError("text install failed")
    if failure == "cache":
        cache_factory.side_effect = RuntimeError("text cache failed")

    def public_load(*_args, **kwargs):
        execution.bind_rpu_execution(model, kwargs["rpu_execution"],
            entry_point="public loader", supported=supported)
        return model

    public_load = Mock(side_effect=public_load)

    class Adapter:
        def __init__(self, raw, *, rpu_execution):
            assert raw is model
            execution.bind_rpu_execution(raw, rpu_execution,
                entry_point="MoE adapter", supported=supported)

        def to_rpu(self, **kwargs):
            return install(**kwargs)

    # Dense imports/constructor cannot accidentally handle this checkpoint.
    wrong_dense = Mock(side_effect=AssertionError("MoE entered dense adapter"))
    for name, attributes in {
        "torch": {"float16": "fp16"},
        "transformers": {"AutoTokenizer": SimpleNamespace(from_pretrained=lambda *_a, **_k: tokenizer)},
        "rpu_backend.model_registry": {"model_path": lambda _: "/cache/qwen35-mixed"},
        "rpu_backend.adapters.qwen3_5": {"Qwen3_5Adapter": wrong_dense},
        "rpu_backend.adapters.qwen3_5_moe": {"Qwen3_5MoeAdapter": Adapter},
        "rpu_backend.api": {
            "RPUModelForConditionalGeneration": SimpleNamespace(from_pretrained=public_load),
            "Qwen3_5MoeCache": SimpleNamespace(from_model=cache_factory),
        },
    }.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    if failure:
        with pytest.raises(RuntimeError, match=f"text {failure} failed"):
            runner.prepare(config)
        model.close.assert_called_once_with()
    else:
        infer, owner = runner.prepare(config)
        assert owner is model and callable(infer)
        infer()
        assert generate.call_args.kwargs["token_selector"] is None
        cache_factory.assert_called_once_with(model, max_seq_len=prefill_length + decode_steps)
    install.assert_called_once_with(max_seq_len=prefill_length + decode_steps)
    public_load.assert_called_once_with(
        "/cache/qwen35-mixed", dtype="fp16", local_files_only=True,
        rpu_execution=request)
    assert model._rpu_execution == request
    wrong_dense.assert_not_called()


@pytest.mark.parametrize("field, value", [
    ("precision", "fp16"),
    ("asset_alias", "qwen3_5-2b"),
    ("component", "multimodal"),
    ("id", "qwen3_5-35b-a3b.mixed-e4m3.text-tp8"),
])
def test_qwen35_moe_text_example_rejects_substituted_profile(monkeypatch, field, value):
    from copy import deepcopy

    common = load(monkeypatch, "example_moe_profile_rejection", "examples/common/config.py")
    profile = deepcopy(common.planner_profiles()["qwen3_5-35b-a3b.mixed-e4m3.text"])
    profile[field] = value
    with pytest.raises(common.ConfigError, match="registered profile"):
        common._validate_execution({"model": {"num_cores": 6}}, profile=profile)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_post_install_cache_failure_retires_owner_and_preserves_primary(monkeypatch, cleanup_fails):
    import torch

    common = load(monkeypatch, "_common", "examples/common/config.py")
    metadata_check = Mock()
    monkeypatch.setattr(common, "validate_checkpoint_profile", metadata_check)
    runner = load(monkeypatch, "example_setup_failure", "examples/common/preparation.py")
    error = RuntimeError("cache setup failed")
    model = SimpleNamespace(model=SimpleNamespace(language_model=object()), close=Mock())
    if cleanup_fails:
        model.close.side_effect = RuntimeError("native cleanup failed")
    processor = SimpleNamespace(tokenizer=lambda *_a, **_kw: {"input_ids": torch.tensor([[1]])})
    for name, attributes in {
        "transformers": {"AutoTokenizer": Mock(), "AutoProcessor": SimpleNamespace(
            from_pretrained=lambda *_a, **_kw: processor)},
        "rpu_backend.model_registry": {"model_path": lambda _alias: "/unused-local-model"},
        "rpu_backend.api": {
            "RPUModelForConditionalGeneration": SimpleNamespace(from_pretrained=lambda *_a, **_kw: model),
            "RPUCache": SimpleNamespace(from_model=Mock(side_effect=error)),
        },
    }.items():
        stub = ModuleType(name)
        vars(stub).update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    with pytest.raises(RuntimeError, match="cache setup failed") as caught:
        runner.prepare({
            "example": {"registry_alias": "qwen3-vl-2b", "target": "qwen3_vl",
                        "profile_id": "qwen3-vl-2b.fp16.text"},
            "input": {"component": "text", "prompt": "hello"}})
    assert caught.value is error
    metadata_check.assert_called_once()
    model.close.assert_called_once_with()
    assert ("native cleanup failed" in " ".join(getattr(error, "__notes__", ()))) is cleanup_fails


def test_qwen3vl_32b_example_reuses_public_quantized_loader_binding(monkeypatch):
    from examples.common import generation
    common = load(monkeypatch, "_common", "examples/common/config.py")
    metadata_check = Mock()
    monkeypatch.setattr(common, "validate_checkpoint_profile", metadata_check)
    runner = load(monkeypatch, "example_32b_binding", "examples/common/preparation.py")
    registry = load(monkeypatch, "example_32b_registry", "python/rpu_backend/model_registry.py")
    config = common.load_config(ROOT / "examples/configs/qwen3_vl/vl/32b/w8a16_text.toml")
    checkpoint = str(registry.model_path(config["example"]["registry_alias"]))
    # The runner records the request geometry before the first forward, so the
    # processor output and the configured images carry their real sizes here.
    ids = SimpleNamespace(shape=(1, 7))
    processor = SimpleNamespace(apply_chat_template=Mock(return_value={"input_ids": ids}),
                                tokenizer=Mock())
    model = SimpleNamespace(model=SimpleNamespace(language_model=object()))
    public_load = Mock(return_value=model)
    monkeypatch.setattr(generation, "_image_inputs",
                        lambda _inputs: [SimpleNamespace(size=(64, 48))])
    for name, attributes in {
        "torch": {"float16": "fp16"},
        "transformers": {"AutoTokenizer": Mock(), "AutoProcessor": SimpleNamespace(
            from_pretrained=Mock(return_value=processor))},
        "rpu_backend.model_registry": {"model_path": registry.model_path},
        "rpu_backend.api": {
            "RPUModelForConditionalGeneration": SimpleNamespace(from_pretrained=public_load),
            "RPUCache": SimpleNamespace(from_model=Mock(return_value=object())),
        },
    }.items():
        stub = ModuleType(name)
        vars(stub).update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    infer, owner = runner.prepare(config)
    assert owner is model and callable(infer)
    metadata_check.assert_called_once_with(config, checkpoint)
    # The original API inspects checkpoint quant_config; FP16 here is activation
    # dtype, and must not divert this checkpoint to a generic FP16 weight loader.
    public_load.assert_called_once_with(
        checkpoint, dtype="fp16", device="rpu", rpu_execution=config["rpu_execution"],
        local_files_only=True, trust_remote_code=False)
    assert config["example"]["dtype"] == "w8a16"
