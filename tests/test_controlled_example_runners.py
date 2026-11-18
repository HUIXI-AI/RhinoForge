"""Host checks for the real public calls behind controlled examples."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def example_entry_path(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))


def module(monkeypatch, name, **values):
    result = ModuleType(name)
    vars(result).update(values)
    monkeypatch.setitem(sys.modules, name, result)
    return result


def config(target):
    return {"example": {"target": target, "registry_alias": "exact-model", "dtype": "w8a16",
                        "opt_in": {"CONTROLLED_OPT_IN": "1"}},
            "input": {"instruction": "move", "proprioception": [0.0]},
            "rpu_execution": {"components": {"action": {"chunk_size": 32}}}}


@pytest.mark.parametrize("target", ["lingbot2", "wall_oss_rtc"])
def test_policy_examples_use_public_api_and_forward_execution(monkeypatch, target):
    from examples.common import preparation
    from examples.runners.lingbot2 import runner as lingbot2
    from examples.runners.wall_oss import rtc
    runner = lingbot2 if target == "lingbot2" else rtc
    value = config(target)
    value["input"]["fp16_registry_alias"] = "exact-fp16"
    policy = Mock()
    policy.to.return_value = policy
    output = object()
    policy.infer.return_value = SimpleNamespace(actions=output, actions_full=output)
    constructor = Mock(return_value=policy)
    arena_prepare = Mock(return_value=True)
    arena_policy = Mock(return_value=SimpleNamespace(prepare_arenas=arena_prepare))
    module(monkeypatch, "rpu_backend.graph",
           GraphRuntimePolicy=SimpleNamespace(from_environment=arena_policy))
    build_order = Mock()
    build_order.attach_mock(arena_prepare, "arenas")
    build_order.attach_mock(constructor, "checkpoint")
    module(monkeypatch, "rpu_backend.api", **{
        "Lingbot2Policy" if target == "lingbot2" else "WallOssPolicy":
            SimpleNamespace(from_checkpoint=constructor)})
    module(monkeypatch, "rpu_backend.model_registry", model_path=lambda alias: f"/cache/{alias}")
    images = [object(), object(), object()]
    monkeypatch.setattr(runner, "_images", lambda inputs: images)

    infer, owner = preparation.prepare(value)

    assert owner is policy
    assert infer() is output
    assert constructor.call_args.kwargs["rpu_execution"] is value["rpu_execution"]
    assert constructor.call_args.kwargs["runtime_env"] == value["example"]["opt_in"]
    assert policy.prepare_graphs.call_args.kwargs["images"] is images
    if target == "lingbot2":
        constructor.assert_called_once_with(
            Path("/cache/exact-model"), dtype="w8a16",
            runtime_env=value["example"]["opt_in"],
            rpu_execution=value["rpu_execution"],
        )
        assert policy.prepare_graphs.call_args.kwargs["prefix_length_range"] == [215, 225]
    else:
        arena_policy.assert_called_once_with(graph_arena_count=15)
        assert [call[0] for call in build_order.mock_calls] == ["arenas", "checkpoint"]
        assert policy.prepare_graphs.call_args.kwargs["prefix_length_range"] == [599, 640]
        assert constructor.call_args.kwargs["rtc"] is True
        assert constructor.call_args.kwargs["fp16_ckpt_dir"] == "/cache/exact-fp16"
        assert policy.infer.call_args.kwargs["use_rtc"] is True


def test_wall_rtc_failed_arena_reservation_prevents_checkpoint_load(monkeypatch):
    from examples.common import preparation
    from examples.runners.wall_oss import rtc

    value = config("wall_oss_rtc")
    value["input"]["fp16_registry_alias"] = "exact-fp16"
    constructor = Mock()
    module(monkeypatch, "rpu_backend.api",
           WallOssPolicy=SimpleNamespace(from_checkpoint=constructor))
    module(monkeypatch, "rpu_backend.model_registry", model_path=Mock())
    reserve = Mock(return_value=False)
    arena_policy = Mock(return_value=SimpleNamespace(prepare_arenas=reserve))
    module(monkeypatch, "rpu_backend.graph",
           GraphRuntimePolicy=SimpleNamespace(from_environment=arena_policy))
    monkeypatch.setattr(rtc, "_images", lambda _: [object(), object(), object()])

    with pytest.raises(RuntimeError, match="could not prepare fifteen Graph arenas"):
        preparation.prepare(value)

    arena_policy.assert_called_once_with(graph_arena_count=15)
    reserve.assert_called_once_with()
    constructor.assert_not_called()


def test_missing_assets_fail_before_model_load(monkeypatch, tmp_path):
    from examples.common import preparation
    constructor = Mock()
    module(monkeypatch, "rpu_backend.api", Lingbot2Policy=SimpleNamespace(from_checkpoint=constructor))
    value = config("lingbot2")
    value["input"] = {"images": [str(tmp_path / "absent.png")], "instruction": "move", "proprioception": [0.0]}
    with pytest.raises(FileNotFoundError, match="absent.png"):
        preparation.prepare(value)
    constructor.assert_not_called()


def test_qwen35_checkpoint_override_cannot_relabel_another_profile(tmp_path):
    from examples.common import inputs
    value = config("qwen3_5_vision")
    value["input"]["checkpoint"] = str(tmp_path)
    with pytest.raises(ValueError, match="only supported for exact Wall RTC/G0.5"):
        inputs._controlled_checkpoint(value)


def test_failed_prepare_closes_partial_policy(monkeypatch):
    from examples.common import preparation
    from examples.runners.lingbot2 import runner
    policy = Mock()

    def fail(runner):
        runner.owner = policy
        raise ValueError("bad fixture")

    monkeypatch.setattr(runner, "prepare", fail)
    with pytest.raises(ValueError, match="bad fixture"):
        preparation.prepare(config("lingbot2"))
    policy.close.assert_called_once_with()


def test_exact_customer_checkpoint_override_does_not_resolve_broad_alias(monkeypatch, tmp_path):
    from examples.common import inputs
    resolve = Mock(side_effect=AssertionError("broad registry alias must not replace exact release"))
    module(monkeypatch, "rpu_backend.model_registry", model_path=resolve)
    value = config("g05")
    value["input"]["checkpoint"] = str(tmp_path)
    assert inputs._controlled_checkpoint(value) == tmp_path
    value["input"]["checkpoint"] = str(tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        inputs._controlled_checkpoint(value)
    resolve.assert_not_called()


def test_g05_uses_exact_loader_and_fresh_batch_for_each_call(monkeypatch, tmp_path):
    from examples.common import preparation
    from examples.runners.g05 import runner
    from contextlib import contextmanager

    value = config("g05")
    value["input"] = {"release_root": str(tmp_path)}
    original = {"samples": [{"value": 1}], "pixel_values": {"camera": 2}}
    monkeypatch.setattr(runner, "_load_mapping", lambda *_: original)
    monkeypatch.setattr(runner, "_checkpoint", lambda _: tmp_path / "checkpoint")
    policy = Mock(_rpu_g05_fm_only_customer_release=True)
    policy.eval.return_value = policy

    def predict(batch):
        assert batch["samples"][0]["value"] == 1
        batch["samples"][0]["value"] = 2
        return {"action": "raw-action"}

    policy.predict_action.side_effect = predict
    patch = Mock()
    module(monkeypatch, "rpu_backend.adapters.g05", patch_g05_policy_for_rpu=patch)

    @contextmanager
    def transaction(root):
        yield lambda: None

    manifest = Mock()
    module(monkeypatch, "rpu_backend.adapters.g05.release_replay",
           _checkpoint_manifest=manifest, _release_import_transaction=transaction,
           _resolve_release_paths=lambda checkpoint, root: (root, checkpoint))
    module(monkeypatch, "vla_rtc.loader", __file__=str(tmp_path / "vla_rtc/loader.py"),
           load_for_eval=lambda *_args, **_kwargs: (policy, object()))

    infer, owner = preparation.prepare(value)

    assert owner is policy
    assert infer() == infer() == "raw-action"
    manifest.assert_called_once_with(tmp_path / "checkpoint")
    assert patch.call_args.kwargs["rpu_execution"] is value["rpu_execution"]
    assert original["samples"][0]["value"] == 1


def test_tensor_fixture_loading_is_weights_only(monkeypatch, tmp_path):
    from examples.common import inputs
    path = tmp_path / "inputs.pt"
    path.touch()
    load = Mock(return_value={"images": "fixture"})
    module(monkeypatch, "torch", load=load)
    assert inputs._load_mapping({"fixture": str(path)}, "fixture") == {"images": "fixture"}
    load.assert_called_once_with(path, map_location="cpu", weights_only=True)


@pytest.mark.parametrize("moe,cores,timing_mode", [
    (False, 8, "per_call"), (False, 8, "legacy_dense_vl"),
    (True, 4, "per_call"), (True, 8, "per_call"),
])
def test_qwen35_vision_reuses_hybrid_cache_and_public_multimodal_forward(
        monkeypatch, moe, cores, timing_mode):
    from examples.common import preparation
    from examples.runners.qwen3_5 import vision as runner
    import torch
    from examples.common import generation as text_runner

    generate = text_runner._generate

    def cpu_generate(*args, **kwargs):
        assert kwargs.pop("input_device") == "cpu"
        return generate(*args, input_device="cpu", **kwargs)

    monkeypatch.setattr(text_runner, "_generate", cpu_generate)
    value = config("qwen3_5_vision")
    value["input"] = {"decode_steps": 2}
    value["run"] = {"timing_mode": timing_mode}
    value["example"]["architecture"] = "Qwen3_5MoeForConditionalGeneration" if moe else "Qwen3_5ForConditionalGeneration"
    if timing_mode == "legacy_dense_vl":
        value["example"].update(dtype="fp16", registry_alias="qwen3_5-2b")
        value["input"].update(prefill_tokens=123, stop_on_eos=False)
    value["rpu_execution"] = {"model": {"num_cores": cores}}
    monkeypatch.setattr(runner, "_images", lambda _: [object()])
    monkeypatch.setattr(runner, "_checkpoint", lambda _: Path("/cache/q35"))
    encoded = {name: Mock() for name in
               ("input_ids", "pixel_values", "image_grid_thw", "mm_token_type_ids")}
    for tensor in encoded.values():
        tensor.to.return_value = tensor
    encoded["input_ids"] = torch.ones((1, 123), dtype=torch.long)
    original_to = torch.Tensor.to

    def no_id_upload(tensor, *args, **kwargs):
        if (args and args[0] == "rpu") or kwargs.get("device") == "rpu":
            pytest.fail("Token IDs were uploaded before host embedding")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", no_id_upload)
    processor = Mock(return_value=encoded)
    module(monkeypatch, "transformers", AutoProcessor=SimpleNamespace(from_pretrained=lambda *_a, **_k: processor))
    model = Mock()
    model.return_value.logits = torch.tensor([[[1., 2.]]])
    model.config.text_config.vocab_size = 2
    constructor = Mock(return_value=model)
    generation_spec = importlib.util.spec_from_file_location(
        "controlled_generation_test", ROOT / "python/rpu_backend/api/generation.py")
    generation = importlib.util.module_from_spec(generation_spec)
    generation_spec.loader.exec_module(generation)
    module(monkeypatch, "rpu_backend.api", RPUModelForConditionalGeneration=SimpleNamespace(from_pretrained=constructor),
           greedy_token_ids=generation.greedy_token_ids)
    cache = Mock()
    cache_constructor = Mock(return_value=cache)
    if moe:
        adapter = Mock()
        adapter.to_rpu.return_value = model
        adapter_constructor = Mock(return_value=adapter)
        module(monkeypatch, "rpu_backend.adapters.qwen3_5_moe", Qwen3_5MoeAdapter=adapter_constructor)
        module(monkeypatch, "rpu_backend.api.qwen3_5_moe_cache", Qwen3_5MoeCache=SimpleNamespace(from_model=cache_constructor))
    else:
        module(monkeypatch, "rpu_backend.api.qwen3_5_cache", Qwen3_5Cache=SimpleNamespace(from_model=cache_constructor))
        if timing_mode == "legacy_dense_vl":
            model.half.return_value = model
            module(monkeypatch, "rpu_backend.adapters.qwen3_5",
                   Qwen3_5Adapter=lambda *_a, **_kw: SimpleNamespace(to_rpu=lambda **_kw: model))
    infer, owner = preparation.prepare(value)
    result = infer()
    infer()

    assert result["text"] is processor.tokenizer.decode.return_value
    assert processor.tokenizer.decode.call_count == 2
    args, kwargs = processor.tokenizer.decode.call_args
    assert torch.equal(args[0], result["token_ids"][0])
    assert kwargs == {"skip_special_tokens": True}
    assert owner is model
    assert cache_constructor.call_args.args[0] is model
    assert cache_constructor.call_args.kwargs["max_seq_len"] == (
        125 if timing_mode == "legacy_dense_vl" else 128)
    if moe:
        assert adapter_constructor.call_args.kwargs["rpu_execution"] is value["rpu_execution"]
        adapter.to_rpu.assert_called_once_with(max_seq_len=128)
    assert cache.reset.call_count == 2
    assert model.call_count == 6
    first = model.call_args_list[0].kwargs
    assert first["image_grid_thw"] is encoded["image_grid_thw"]
    assert first["mm_token_type_ids"] is encoded["mm_token_type_ids"]
    assert first["past_key_values"] is cache
    assert first["input_ids"] is encoded["input_ids"]
    assert all(call.kwargs["input_ids"].device.type == "cpu" for call in model.call_args_list)
    assert constructor.call_args.kwargs["rpu_execution"] is value["rpu_execution"]


def test_halo_single_step_keeps_captured_prefix_and_planner(monkeypatch, tmp_path):
    import torch
    from examples.common import preparation
    from examples.runners.halo import runner
    value = config("halo")
    checkpoint = tmp_path / "HALO/action_expert.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    monkeypatch.setattr(runner, "_checkpoint", lambda _: checkpoint)
    key, val = torch.zeros(3, 2, 128), torch.ones(3, 2, 128)
    fixture = {"packed_position_ids": torch.full((18,), 11),
               "packed_text_indexes": torch.tensor([0, 17]),
               "packed_action_token_indexes": torch.arange(1, 17),
               "key_values_lens": torch.tensor([3]),
               "past_key_values": {"key_cache": {0: key}, "value_cache": {0: val}},
               "x_t": torch.zeros(16, 14), "timestep": torch.ones(16),
               "packed_action_position_ids": torch.arange(16), "packed_text_ids": torch.tensor([1, 2])}
    value["input"]["fixture"] = str(tmp_path / "step.pt")
    torch.save({"kwargs": fixture}, value["input"]["fixture"])
    policy = Mock()
    builder = Mock(return_value=policy)
    module(monkeypatch, "rpu_backend.adapters.halo",
           HaloConfig=lambda: SimpleNamespace(text_rows=[0, 17], action_rows=range(1, 17),
                                             prefix_len=3, num_layers=1, action_dim=14,
                                             chunk_size=18, num_kv_heads=2, head_dim=128),
           build_halo_action_expert=builder)

    infer, owner = preparation.prepare(value)
    infer()

    assert owner is policy
    policy.insert_prefix.assert_called_once()
    assert torch.equal(policy.insert_prefix.call_args.args[0][0], key)
    assert torch.equal(policy.insert_prefix.call_args.args[1][0], val)
    policy.step.assert_called_once()
    for actual, name in zip(policy.step.call_args.args,
                            ("x_t", "timestep", "packed_action_position_ids", "packed_text_ids")):
        assert torch.equal(actual, fixture[name])
    assert builder.call_args.args == (checkpoint,)
    assert builder.call_args.kwargs["rpu_execution"] is value["rpu_execution"]


def test_rhinovla_uses_pinned_factory_and_public_predict(monkeypatch, tmp_path):
    from examples.common import preparation
    from examples.common import inputs
    from examples.runners.rhinovla import runner
    value = config("rhinovla")
    value["input"] = {"trust_model_code": True, "num_steps": 10}
    fixture = {name: Mock() for name in ("noise", "state", "state_mask", "action_mask")}
    fixture.update(qwen_inputs={"input_ids": "ids"}, metadata={
        "kind": "rhinovla_official_cpu_fp32_nonrtc", "source_commit": "source-sha",
        "checkpoint_sha256": "checkpoint-sha", "rtc": False})
    monkeypatch.setattr(inputs, "_controlled_checkpoint", lambda _: tmp_path)
    monkeypatch.setattr(runner, "_load_mapping", lambda *_: fixture)
    policy = Mock()
    constructor = Mock(return_value=policy)
    module(monkeypatch, "rpu_backend.api", RhinoVLAPolicy=SimpleNamespace(from_factory=constructor))
    module(monkeypatch, "rpu_backend.adapters.rhinovla.factory",
           OFFICIAL_SOURCE_SHA="source-sha", OFFICIAL_CHECKPOINT_SHA256="checkpoint-sha")

    infer, owner = preparation.prepare(value)
    infer()
    infer()

    assert owner is policy and policy.predict.call_count == 2
    name, params = constructor.call_args.args
    assert name == "rpu_backend.adapters.rhinovla.factory:official_runtime_factory"
    assert params["model_root"] == tmp_path / "source"
    assert params["checkpoint"] == tmp_path / "weights/rhinovla_pretrain.ckpt"
    assert params["qin"] is fixture["qwen_inputs"] and params["steps"] == 10
    assert constructor.call_args.kwargs["rpu_execution"] is value["rpu_execution"]
    assert policy.predict.call_args.kwargs["x0"] is fixture["noise"]
    constructor.reset_mock()
    fixture["metadata"]["rtc"] = True
    with pytest.raises(ValueError, match="pinned official non-RTC"):
        preparation.prepare(value)
    constructor.assert_not_called()
    value["input"]["trust_model_code"] = False
    with pytest.raises(PermissionError, match="trust_model_code"):
        preparation.prepare(value)
    constructor.assert_not_called()


def test_rhinovla_v3_source_raw_images_use_the_source_processor(monkeypatch, tmp_path):
    import torch
    from PIL import Image
    from examples.common import preparation
    from examples.runners.rhinovla import runner

    paths = []
    for index, size in enumerate(((640, 480), (320, 240))):
        paths.append(str(tmp_path / f"{index}.png"))
        Image.new("RGB", size, (index, 0, 0)).save(paths[-1])
    value = config("rhinovla")
    value["example"]["dtype"] = "fp16"
    value["example"]["opt_in"] = {"RPU_RHINOVLA_EXPERT_W8A16": "0", "RPU_RHINOVLA_FULL_W8A16": "0"}
    value["input"] = {"source_dir": "/source", "checkpoint": "/source/model.pt", "trust_model_code": True,
                      "images": paths, "instruction": "pick up the object", "state": [0.5, -0.5, 0.25]}
    value["run"] = {"seed": 0}
    qin = {"input_ids": torch.ones(1, 140, dtype=torch.long), "attention_mask": torch.ones(1, 140),
           "pixel_values": torch.zeros(512, 1536), "image_grid_thw": torch.tensor([[1, 16, 16]] * 2)}
    policy = Mock()
    policy.runtime = SimpleNamespace(qin=qin)
    constructor = Mock(return_value=policy)
    module(monkeypatch, "rpu_backend.api", RhinoVLAPolicy=SimpleNamespace(from_factory=constructor))
    module(monkeypatch, "rpu_backend.adapters.rhinovla.factory",
           V3_SOURCE_SHA="source-sha", V3_CHECKPOINT_SHA256="checkpoint-sha")
    monkeypatch.setattr(runner, "_confirm_v3_source_quantization", lambda *_: None)

    infer, owner = preparation.prepare(value)
    infer()

    name, params = constructor.call_args.args
    assert owner is policy and name == "rpu_backend.adapters.rhinovla.factory:v3_runtime_factory"
    raw = params["raw_inputs"]
    assert "qin" not in params and raw["view_roles"] == ["top_head", "hand_left"]
    assert [image.size for image in raw["images"]] == [(256, 256)] * 2
    assert {image.mode for image in raw["images"]} == {"RGB"}
    request = policy.predict.call_args.kwargs
    mask = torch.zeros(1, 96)
    mask[:, :3] = 1
    assert torch.equal(request["state_mask"], mask) and torch.equal(request["action_mask"], mask)
    assert request["state"][0, :4].tolist() == [0.5, -0.5, 0.25, 0.0]
    x0 = torch.randn(1, 30, 96, generator=torch.Generator().manual_seed(0))
    assert torch.equal(request["x0"], x0 * mask[:, None])
    assert infer.input_geometry["prefix_tokens"] == 140 and infer.input_geometry["cameras"] == 2
    # measurement.run() materializes the geometry into the report right after prepare.
    assert infer.input_geometry["image_grid_thw"].tolist() == [[1, 16, 16]] * 2
    value["input"]["fixture"] = "/fixture.pt"
    with pytest.raises(ValueError, match="exactly one of"):
        preparation.prepare(value)
