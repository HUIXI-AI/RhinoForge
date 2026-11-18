"""Host-only checks for the public policy paths used by model examples."""

from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def example_entry_path(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))


def module(monkeypatch, name, **values):
    result = ModuleType(name)
    result.__path__ = []
    vars(result).update(values)
    monkeypatch.setitem(sys.modules, name, result)
    return result


def backend_modules(monkeypatch):
    module(monkeypatch, "rpu_backend")
    module(monkeypatch, "rpu_backend.adapters")


def config(target, dtype="fp16"):
    return {
        "example": {
            "profile_id": "registered-profile",
            "registry_alias": "registered-model",
            "target": target,
            "dtype": dtype,
            "opt_in": {},
        },
        "input": {},
        "rpu_execution": {"components": {"action": {"chunk_size": 32}}},
    }


@pytest.mark.parametrize(
    ("dtype", "w8a16", "w4a16", "needs_fp16"),
    [
        ("fp16", False, False, False),
        ("w8a16", True, False, True),
        ("w4a16", False, True, True),
        ("w4a16-pgrp", False, True, True),
    ],
)
def test_wall_oss_uses_rc9_public_policy_and_closable_runtime(
    monkeypatch, dtype, w8a16, w4a16, needs_fp16
):
    from examples.common import preparation
    from examples.runners.wall_oss import runner
    backend_modules(monkeypatch)
    runtime = object()
    output = object()
    policy = Mock(_vla=runtime)
    policy.to.return_value = policy
    policy.infer.return_value = SimpleNamespace(actions=output)
    constructor = Mock(return_value=policy)
    module(monkeypatch, "rpu_backend.api", WallOssPolicy=SimpleNamespace(from_checkpoint=constructor))
    value = config("wall_oss", dtype)
    value["input"] = {
        "instruction": "move",
        "proprioception": [0.0] * 26,
        "state_mask": [1] * 7 + [0] * 19,
        "camera_names": ["face_view", "left_wrist_view", "right_wrist_view"],
        "prefix_length_range": [545, 640],
        "noise_seed": 7,
    }
    images = [object(), object(), object()]
    monkeypatch.setattr(runner, "_images", lambda _: images)
    monkeypatch.setattr(runner, "_checkpoint", lambda _: Path(f"/cache/wall-{dtype}"))
    model_path = Mock(return_value=Path("/cache/wall-fp16"))
    monkeypatch.setattr(runner, "_model_path", model_path)
    if needs_fp16:
        value["input"]["fp16_registry_alias"] = "wall-fp16"

    infer, owner = preparation.prepare(value)

    assert owner is runtime
    assert infer() is output
    kwargs = constructor.call_args.kwargs
    assert kwargs["dataset_key"] == "berkeley_autolab_ur5"
    assert kwargs["camera_names"] == tuple(value["input"]["camera_names"])
    assert kwargs["w8a16"] is w8a16
    assert kwargs["w4a16"] is w4a16
    assert kwargs["fp16_ckpt_dir"] == ("/cache/wall-fp16" if needs_fp16 else None)
    assert kwargs["rpu_execution"] is value["rpu_execution"]
    policy.prepare_graphs.assert_called_once()
    prepared = policy.prepare_graphs.call_args.kwargs
    assert prepared["prefix_length_range"] == [545, 640]
    assert prepared["state_mask"] == value["input"]["state_mask"]
    assert policy.infer.call_args.kwargs["images"] is images


@pytest.mark.parametrize("dtype,inherited,failure", [
    ("fp16", "1", False), ("w8a16", "0", False),
    ("fp16", None, False), ("w8a16", None, True),
])
@pytest.mark.parametrize("num_steps", [4, 10])
def test_gr00t_uses_registered_backbone_and_preprocessed_observation(monkeypatch, dtype, inherited, failure, num_steps):
    from examples.common import preparation
    from examples.runners.gr00t import runner
    backend_modules(monkeypatch)
    runtime = Mock()
    runtime.get_action.return_value = "normalized-action"
    if inherited is None:
        monkeypatch.delenv("RPU_GR00T_W8A16", raising=False)
    else:
        monkeypatch.setenv("RPU_GR00T_W8A16", inherited)
    def build(*args, **kwargs):
        assert runner.os.environ["RPU_GR00T_W8A16"] == ("1" if dtype == "w8a16" else "0")
        if failure:
            raise RuntimeError("injected cold build failure")
        return runtime
    builder = Mock(side_effect=build)
    module(monkeypatch, "rpu_backend.adapters.gr00t", build_gr00t_vla=builder)
    value = config("gr00t", dtype)
    value["input"] = {
        "qwen3vl_registry_alias": "qwen3-vl-2b",
        "embodiment_id": 20,
        "seed": 3,
        "num_steps": num_steps,
    }
    observation = {name: object() for name in (
        "input_ids", "attention_mask", "pixel_values", "image_grid_thw", "state"
    )}
    observation["input_ids"] = SimpleNamespace(shape=(1, 82))
    observation["state"] = SimpleNamespace(shape=(1, 1, 132))
    monkeypatch.setattr(runner, "_load_mapping", lambda *_: observation)
    monkeypatch.setattr(runner, "_checkpoint", lambda _: Path("/cache/gr00t"))
    monkeypatch.setattr(runner, "_model_path", lambda *_args, **_kwargs: Path("/cache/qwen3-vl"))

    if failure:
        with pytest.raises(RuntimeError, match="injected cold build"):
            preparation.prepare(value)
        assert runner.os.environ.get("RPU_GR00T_W8A16") == inherited
        return
    infer, owner = preparation.prepare(value)
    assert runner.os.environ.get("RPU_GR00T_W8A16") == inherited

    assert owner is runtime
    assert infer() == "normalized-action"
    builder.assert_called_once_with(
        "/cache/gr00t", "/cache/qwen3-vl", embodiment_id=20, num_steps=num_steps,
        max_prefix_len=256, rpu_execution=value["rpu_execution"],
    )
    assert runtime.get_action.call_args.args[0] == observation
    assert runtime.get_action.call_args.args[0] is not observation
    assert runtime.get_action.call_args.kwargs == {"seed": 3, "num_steps": num_steps}


@pytest.mark.parametrize("rows,explicit,capacity", [
    (82, None, 256), (193, None, 256), (225, None, 256), (238, None, 256),
    (270, None, 272), (289, None, 304), (334, None, 336), (512, None, 512),
    (193, 208, 208), (334, 512, 512),
])
def test_gr00t_example_sizes_prefix_cache_from_actual_observation(monkeypatch, rows, explicit, capacity):
    from examples.common import preparation
    from examples.runners.gr00t import runner
    backend_modules(monkeypatch)
    owner = Mock()
    builder = Mock(return_value=owner)
    module(monkeypatch, "rpu_backend.adapters.gr00t", build_gr00t_vla=builder)
    value = config("gr00t")
    value["input"] = {"qwen3vl_registry_alias": "qwen3-vl-2b"}
    if explicit is not None:
        value["input"]["max_prefix_len"] = explicit
    observation = {name: object() for name in (
        "attention_mask", "pixel_values", "image_grid_thw", "state")}
    observation["input_ids"] = SimpleNamespace(shape=(1, rows))
    observation["state"] = SimpleNamespace(shape=(1, 1, 132))
    monkeypatch.setattr(runner, "_load_mapping", lambda *_: observation)
    monkeypatch.setattr(runner, "_checkpoint", lambda _: Path("/cache/gr00t"))
    monkeypatch.setattr(runner, "_model_path", lambda *_args, **_kwargs: Path("/cache/qwen3-vl"))
    infer, actual = preparation.prepare(value)
    assert actual is owner and builder.call_args.kwargs["max_prefix_len"] == capacity
    assert builder.call_args.kwargs["num_steps"] == 4
    infer()
    assert owner.get_action.call_args.args[0]["input_ids"].shape[-1] == rows


@pytest.mark.parametrize("rows,capacity", [(334, 256), (334, 335), (334, 528),
    (82, True), (82, 256.0), (82, "256"), (82, 0), (82, -16), (513, None)])
def test_gr00t_example_rejects_bad_prefix_capacity_before_builder(monkeypatch, rows, capacity):
    from examples.common import preparation
    from examples.runners.gr00t import runner
    backend_modules(monkeypatch)
    builder = Mock()
    module(monkeypatch, "rpu_backend.adapters.gr00t", build_gr00t_vla=builder)
    value = config("gr00t")
    if capacity is not None:
        value["input"]["max_prefix_len"] = capacity
    observation = {name: object() for name in (
        "attention_mask", "pixel_values", "image_grid_thw", "state")}
    observation["input_ids"] = SimpleNamespace(shape=(1, rows))
    monkeypatch.setattr(runner, "_load_mapping", lambda *_: observation)
    with pytest.raises(ValueError, match="max_prefix_len"):
        preparation.prepare(value)
    builder.assert_not_called()


@pytest.mark.parametrize("dtype", [None, "nvfp4a16", "unknown"])
def test_gr00t_rejects_unknown_example_precision_before_loading(monkeypatch, dtype):
    from examples.common import preparation
    from examples.runners.gr00t import runner
    backend_modules(monkeypatch)
    builder, load = Mock(), Mock()
    module(monkeypatch, "rpu_backend.adapters.gr00t", build_gr00t_vla=builder)
    monkeypatch.setattr(runner, "_load_mapping", load)
    with pytest.raises(ValueError, match="gr00t runner precision .* names 0 recipes"):
        preparation.prepare(config("gr00t", dtype))
    builder.assert_not_called()
    load.assert_not_called()


def test_hy_vla_returns_safe_normalized_output(monkeypatch):
    from examples.common import preparation
    from examples.runners.hy_vla import runner
    backend_modules(monkeypatch)
    policy = Mock()
    policy.to.return_value = policy
    normalized = object()
    policy.infer.return_value = SimpleNamespace(actions_normalized=normalized)
    constructor = Mock(return_value=policy)
    module(monkeypatch, "rpu_backend.api", HyEmbodiedPolicy=SimpleNamespace(from_checkpoint=constructor))
    value = config("hy_vla")
    value["input"] = {
        "instruction": "move",
        "state": [0.0] * 32,
        "noise_seed": 5,
        "prefix_len": 240,
    }
    images = [object(), object(), object()]
    monkeypatch.setattr(runner, "_images", lambda _: images)
    monkeypatch.setattr(runner, "_checkpoint", lambda _: Path("/cache/hy"))

    infer, owner = preparation.prepare(value)

    assert owner is policy
    assert infer() is normalized
    assert constructor.call_args.kwargs["rpu_execution"] is value["rpu_execution"]
    assert constructor.call_args.kwargs["prefix_len"] == 240
    assert policy.infer.call_args.kwargs == {
        "images": images,
        "instruction": "move",
        "state": value["input"]["state"],
        "noise_seed": 5,
    }


def test_lingbot_v1_uses_public_builder_and_closable_runtime(monkeypatch):
    from examples.common import preparation
    from examples.runners.lingbot_v1 import runner
    backend_modules(monkeypatch)
    runtime = Mock()
    runtime.get_action.return_value = "action"
    builder = Mock(return_value=runtime)
    module(monkeypatch, "rpu_backend.adapters.lingbot_vla", build_lingbot_vla=builder)
    value = config("lingbot", "cpu-fp32+rpu-fp16")
    value["input"] = {"num_steps": 10, "max_seq": 256}
    observation = {"images": object(), "lang_tokens": object(), "state": object(), "noise": object()}
    monkeypatch.setattr(runner, "_load_mapping", lambda *_: observation)
    monkeypatch.setattr(runner, "_checkpoint", lambda _: Path("/cache/lingbot"))

    infer, owner = preparation.prepare(value)

    assert owner is runtime
    assert infer() == "action"
    builder.assert_called_once_with(
        "/cache/lingbot", qwen_base=None, max_seq=256,
        rpu_execution=value["rpu_execution"],
    )
    assert runtime.get_action.call_args.args[0] == observation
    assert runtime.get_action.call_args.args[0] is not observation


@pytest.mark.parametrize("arena_ready", (True, False))
def test_internvla_runs_exact_hash_bound_full_chain(monkeypatch, tmp_path, arena_ready):
    from examples.common import preparation
    from examples.runners.internvla import runner
    import torch

    backend_modules(monkeypatch)
    module(monkeypatch, "rpu_backend.adapters.internvla_n1")
    navdp_path = tmp_path / "navdp_head.safetensors"
    navdp_path.touch()
    frontend_path = tmp_path / "s2_frontend.safetensors"
    frontend_path.touch()
    fixture_path = tmp_path / "fixture.pt"
    fixture_path.touch()
    manifest_path = tmp_path / "assets.json"
    manifest_path.write_text(
        '{"navdp_head.safetensors": "' + "a" * 64 + '", '
        '"s2_frontend.safetensors": "' + "b" * 64 + '", '
        '"fixture.pt": "' + "c" * 64 + '"}',
        encoding="utf-8",
    )
    fixture = {
        "pixel_values": torch.zeros(784, 1),
        "grid_thw": torch.tensor([[1, 28, 28]]),
        "output_ids": torch.zeros((1, 196), dtype=torch.long),
        "image_idx": torch.ones((1, 196), dtype=torch.bool),
        "position_ids": torch.zeros((200, 3), dtype=torch.int64),
        "rgb": torch.empty((1, 2, 224, 224, 3), device="meta"),
        "depth": torch.empty((1, 2, 224, 224, 1), device="meta"),
        "init_noise": torch.empty((4, 32, 3), device="meta"),
    }
    embed_weight = torch.empty((151668, 3584), device="meta")
    latent_queries = torch.empty((1, 4, 3584), device="meta")

    vision = Mock()
    vision.forward.return_value = torch.zeros((196, 3584))
    backbone = Mock()
    backbone.forward_embeds.return_value = torch.zeros((1, 200, 3584))
    navdp = Mock()
    navdp.predict_action.return_value = torch.zeros((4, 32, 3))
    goal_glue = Mock(return_value=torch.zeros((1, 1, 384)))
    rgb_tower, depth_tower, former = Mock(), Mock(), Mock()
    rgbd = Mock(_rpu_towers=(rgb_tower, depth_tower), _rpu_former=former)
    rgbd.return_value = torch.zeros((1, 32, 384))

    vision_builder = Mock(return_value=vision)
    backbone_builder = Mock(return_value=backbone)
    navdp_builder = Mock(return_value=navdp)
    goal_builder = Mock(return_value=goal_glue)
    rgbd_builder = Mock(return_value=rgbd)
    arena_prepare = Mock(return_value=arena_ready)
    arena_policy = Mock(return_value=SimpleNamespace(prepare_arenas=arena_prepare))
    module(monkeypatch, "rpu_backend.graph",
           GraphRuntimePolicy=SimpleNamespace(from_environment=arena_policy))
    build_order = Mock()
    for name, call in (("arenas", arena_prepare), ("vision", vision_builder),
                       ("backbone", backbone_builder), ("navdp", navdp_builder),
                       ("goal", goal_builder), ("rgbd", rgbd_builder)):
        build_order.attach_mock(call, name)
    verify = Mock()
    module(
        monkeypatch,
        "rpu_backend.adapters.internvla_n1._policy",
        require_controlled_environment=Mock(),
        require_controlled_evaluation=Mock(),
        verify_asset_manifest=verify,
    )
    module(monkeypatch, "rpu_backend.adapters.internvla_n1.backbone",
           build_qwen25vl_backbone=backbone_builder)
    module(monkeypatch, "rpu_backend.adapters.internvla_n1.goal_glue",
           build_goal_glue=goal_builder)
    module(monkeypatch, "rpu_backend.adapters.internvla_n1.rgbd_rpu",
           build_rgbd_encoder_rpu=rgbd_builder)
    module(monkeypatch, "rpu_backend.adapters.internvla_n1.vision",
           build_internvla_vision=vision_builder)
    module(monkeypatch, "rpu_backend.adapters.navdp", build_navdp=navdp_builder)

    child_execution = {
        "system2_vision": {"vision": {"chunk_size": "auto"}},
        "system2_backbone": {"prefill": {"chunk_size": "auto"}},
        "navdp_action": {"action": {"chunk_size": "auto"}},
    }
    session = Mock()
    session.shutdown.side_effect = lambda retire: retire()
    controller = SimpleNamespace(execute=lambda: nullcontext(), session=session)
    bind = Mock(return_value=controller)
    own_child = Mock(side_effect=lambda owner, child: child)
    close_runtime = Mock()
    module(
        monkeypatch,
        "rpu_backend.adapters.internvla_n1.execution",
        BACKBONE_COMPONENT="system2_backbone",
        FORMER_COMPONENT="rgbd_former",
        NAVDP_COMPONENT="navdp_action",
        VISION_COMPONENT="system2_vision",
        bind_internvla_execution_runtime=bind,
        own_internvla_child=own_child,
        close_internvla_runtime=close_runtime,
        gc_close_internvla_runtime=Mock(),
        resolve_internvla_execution=lambda execution, **_kwargs: (execution, child_execution),
    )
    value = config("internvla_n1")
    value["input"] = {
        "fixture": str(fixture_path),
        "asset_manifest": str(manifest_path),
        "sample_num": 4,
        "seed": 1,
    }
    monkeypatch.setattr(runner, "_load_mapping", lambda *_: fixture)
    monkeypatch.setattr(runner, "_checkpoint", lambda _: tmp_path)
    monkeypatch.setattr(
        runner, "_load_internvla_frontend",
        lambda _: (embed_weight, latent_queries),
    )
    monkeypatch.delenv("RPU_INTERNVLA_N1_ALLOW_NUMERIC_BLOCKED", raising=False)
    monkeypatch.delenv("RPU_S2_SDPA_BF16", raising=False)

    if not arena_ready:
        with pytest.raises(RuntimeError, match="could not prepare seven Graph arenas"):
            preparation.prepare(value)
        arena_policy.assert_called_once_with(graph_arena_count=7)
        arena_prepare.assert_called_once_with()
        assert [call[0] for call in build_order.mock_calls] == ["arenas"]
        return

    infer, owner = preparation.prepare(value)
    arena_policy.assert_called_once_with(graph_arena_count=7)
    arena_prepare.assert_called_once_with()
    assert [call[0] for call in build_order.mock_calls] == [
        "arenas", "vision", "backbone", "navdp", "goal", "rgbd"]

    action = infer()
    manifest = {
        navdp_path.resolve(): "a" * 64,
        frontend_path.resolve(): "b" * 64,
        fixture_path.resolve(): "c" * 64,
    }
    assert tuple(action.shape) == (4, 32, 3)
    assert owner.vision is vision
    assert owner.backbone is backbone
    verify.assert_called_once_with(
        (navdp_path, frontend_path, fixture_path.resolve()), manifest
    )
    vision_builder.assert_called_once_with(
        tmp_path / "s2_vit",
        asset_manifest=manifest,
        rpu_execution=child_execution["system2_vision"],
    )
    backbone_builder.assert_called_once_with(
        tmp_path / "s2_w8a16",
        asset_manifest=manifest,
        rpu_execution=child_execution["system2_backbone"],
    )
    navdp_builder.assert_called_once_with(
        navdp_path,
        asset_manifest=manifest,
        _defer_execution_session=True,
        rpu_execution=child_execution["navdp_action"],
    )
    assert rgbd_builder.call_args.kwargs["rpu_execution"] is value["rpu_execution"]
    assert bind.call_args.kwargs["backbone"] is backbone
    assert bind.call_args.kwargs["vision"] is vision
    assert bind.call_args.kwargs["navdp"] is navdp
    navdp.predict_action.assert_called_once_with(
        goal_glue.return_value.float(), rgbd.return_value.float(),
        sample_num=4, seed=1, init_noise=fixture["init_noise"].float(),
    )
    owner.close()
    close_runtime.assert_called_once_with(owner)
    assert [call.args for call in own_child.call_args_list] == [
        (owner, child) for child in (vision, backbone, navdp, rgb_tower, depth_tower, former)]


def test_failed_prepare_destroys_partial_owner(monkeypatch):
    from examples.common import preparation
    from examples.runners.gr00t import runner
    value = config("gr00t")
    owner = SimpleNamespace(destroy=Mock())

    def fail(runner):
        runner.owner = owner
        raise ValueError("bad fixture")

    monkeypatch.setattr(runner, "prepare", fail)
    with pytest.raises(ValueError, match="bad fixture"):
        preparation.prepare(value)
    owner.destroy.assert_called_once_with()


def test_tensor_input_is_weights_only_and_runner_has_no_dynamic_factory(monkeypatch, tmp_path):
    from examples.common import inputs
    path = tmp_path / "inputs.pt"
    path.touch()
    load = Mock(return_value={"input_ids": object()})
    module(monkeypatch, "torch", load=load)

    assert inputs._load_mapping({"observation": str(path)}, "observation") == {
        "input_ids": load.return_value["input_ids"]
    }
    load.assert_called_once_with(path, map_location="cpu", weights_only=True)
    source = Path(inputs.__file__).read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "from tests" not in source
