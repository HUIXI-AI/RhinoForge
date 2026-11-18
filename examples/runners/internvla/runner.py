"""Public internvla model construction and input binding."""
from __future__ import annotations

import json
import os
from pathlib import Path

from examples.common.inputs import _required_file, _load_mapping, _policy_checkpoint as _checkpoint

def _load_manifest(path):
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid asset manifest JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("asset manifest JSON must contain a path-to-SHA256 object")
    result = {}
    for value, digest in raw.items():
        asset = Path(os.path.expandvars(value)).expanduser()
        if not asset.is_absolute():
            asset = path.parent / asset
        asset = asset.resolve()
        if asset in result and result[asset] != digest:
            raise ValueError(f"conflicting SHA-256 values for {asset}")
        result[asset] = digest
    return result

class _InternVLAOwner:
    """One directly closable owner for the controlled multi-runtime bundle."""

    def __init__(self):
        self._internvla_execution_controller = None

    def own(self, resource):
        from rpu_backend.adapters.internvla_n1.execution import own_internvla_child
        return own_internvla_child(self, resource)

    def close(self):
        from rpu_backend.adapters.internvla_n1.execution import close_internvla_runtime
        close_internvla_runtime(self)

    def __del__(self):
        from rpu_backend.adapters.internvla_n1.execution import gc_close_internvla_runtime
        gc_close_internvla_runtime(self)

def _load_internvla_frontend(path):
    from safetensors.torch import load_file

    frontend = load_file(str(path), device="cpu")
    embed_weight = frontend.get("model.embed_tokens.weight")
    latent_queries = frontend.get("model.latent_queries")
    if embed_weight is None or tuple(embed_weight.shape) != (151668, 3584):
        raise ValueError("InternVLA-N1 frontend embed must be [151668,3584]")
    if latent_queries is None or tuple(latent_queries.shape) != (1, 4, 3584):
        raise ValueError("InternVLA-N1 latent queries must be [1,4,3584]")
    return embed_weight.float(), latent_queries.float()

def _internvla_n1(runner):
    config, inputs = runner.source_config, runner.source_config["input"]
    if os.environ.get("RPU_S2_SDPA_BF16") not in (None, "", "0"):
        raise ValueError("InternVLA-N1 controlled profile requires RPU_S2_SDPA_BF16=0")

    from rpu_backend.adapters.internvla_n1._policy import (
        require_controlled_environment,
        require_controlled_evaluation,
        verify_asset_manifest,
    )
    require_controlled_environment()
    manifest = _load_manifest(_required_file(inputs, "asset_manifest"))
    require_controlled_evaluation(manifest)
    root = _checkpoint(config).resolve()
    fixture_path = _required_file(inputs, "fixture").resolve()
    navdp_path = root / "navdp_head.safetensors"
    frontend_path = root / "s2_frontend.safetensors"
    verify_asset_manifest((navdp_path, frontend_path, fixture_path), manifest)
    fixture = _load_mapping(inputs, "fixture")
    required = {
        "pixel_values", "grid_thw", "output_ids", "image_idx",
        "position_ids", "rgb", "depth", "init_noise",
    }
    missing = sorted(required.difference(fixture))
    if missing:
        raise ValueError(f"InternVLA-N1 full-chain fixture missing fields: {missing}")

    import torch

    embed_weight, latent_queries = _load_internvla_frontend(frontend_path)
    output_ids = fixture["output_ids"]
    image_idx = fixture["image_idx"]
    if output_ids.ndim != 2 or output_ids.shape[0] != 1:
        raise ValueError("InternVLA-N1 output_ids must be [1,S]")
    if tuple(image_idx.shape) != tuple(output_ids.shape):
        raise ValueError("InternVLA-N1 image_idx must match output_ids")
    if output_ids.numel() and (
        int(output_ids.min()) < 0 or int(output_ids.max()) >= embed_weight.shape[0]
    ):
        raise ValueError("InternVLA-N1 output_ids exceed the exact vocabulary")
    position_ids = fixture["position_ids"]
    if tuple(position_ids.shape[:2]) == (3, 1):
        position_ids = position_ids[:, 0, :].t().contiguous()
    if tuple(position_ids.shape) != (output_ids.shape[1] + 4, 3):
        raise ValueError("InternVLA-N1 position_ids must cover text plus 4 latent rows")
    grid_thw = fixture["grid_thw"].to(torch.int64)
    if tuple(grid_thw.shape) != (1, 3) or tuple(grid_thw[0].tolist()) != (1, 28, 28):
        raise ValueError("InternVLA-N1 exact Vision grid must be [[1,28,28]]")
    if fixture["pixel_values"].shape[0] != 784 or int(image_idx.sum()) != 196:
        raise ValueError("InternVLA-N1 exact Vision requires 784 patches / 196 image tokens")
    exact_shapes = {
        "rgb": (1, 2, 224, 224, 3),
        "depth": (1, 2, 224, 224, 1),
        "init_noise": (4, 32, 3),
    }
    bad_shapes = {
        name: tuple(fixture[name].shape)
        for name, shape in exact_shapes.items()
        if tuple(fixture[name].shape) != shape
    }
    if bad_shapes:
        raise ValueError(f"InternVLA-N1 full-chain fixture shape mismatch: {bad_shapes}")
    sample_num = inputs.get("sample_num", 4)
    if sample_num != fixture["init_noise"].shape[0]:
        raise ValueError("InternVLA-N1 sample_num must match fixture init_noise")

    from rpu_backend.adapters.internvla_n1.backbone import build_qwen25vl_backbone
    from rpu_backend.adapters.internvla_n1.execution import (
        BACKBONE_COMPONENT,
        NAVDP_COMPONENT,
        VISION_COMPONENT,
        bind_internvla_execution_runtime,
        resolve_internvla_execution,
    )
    from rpu_backend.adapters.internvla_n1.goal_glue import build_goal_glue
    from rpu_backend.adapters.internvla_n1.rgbd_rpu import build_rgbd_encoder_rpu
    from rpu_backend.adapters.internvla_n1.vision import build_internvla_vision
    from rpu_backend.adapters.navdp import build_navdp

    execution, children = resolve_internvla_execution(
        config.get("rpu_execution"), entry_point="InternVLA-N1 example"
    )
    from rpu_backend.graph import GraphRuntimePolicy
    # Six persistent component graphs and one transient patch/eager queue.
    if not GraphRuntimePolicy.from_environment(graph_arena_count=7).prepare_arenas():
        raise RuntimeError("InternVLA-N1 could not prepare seven Graph arenas")
    owner = _InternVLAOwner()
    runner.owner = owner
    owner.vision = owner.own(build_internvla_vision(
        root / "s2_vit", asset_manifest=manifest,
        rpu_execution=children[VISION_COMPONENT]))
    owner.backbone = owner.own(build_qwen25vl_backbone(
        root / "s2_w8a16", asset_manifest=manifest,
        rpu_execution=children[BACKBONE_COMPONENT]))
    owner.navdp = owner.own(build_navdp(
        navdp_path, asset_manifest=manifest,
        _defer_execution_session=True,
        rpu_execution=children[NAVDP_COMPONENT]))
    owner.goal_glue = build_goal_glue(navdp_path, device="cpu")
    owner.rgbd = build_rgbd_encoder_rpu(
        navdp_path, root / "rgbd_src", asset_manifest=manifest,
        rpu_execution=execution,
    )
    rgb_tower, depth_tower = owner.rgbd._rpu_towers
    owner.own(rgb_tower)
    owner.own(depth_tower)
    owner.own(owner.rgbd._rpu_former)
    # System 2 (Vision, backbone prefill) and System 1 (RGB-D encoder plus NavDP head).
    runner.stage_boundaries = tuple((stage, getattr(owner, part, None), name) for stage, part, name in (
        ("vision", "vision", "forward"), ("prefill", "backbone", "forward_embeds"),
        ("navdp", "rgbd", "forward"), ("navdp", "navdp", "predict_action")))
    owner._internvla_execution_controller = bind_internvla_execution_runtime(
        owner,
        backbone=owner.backbone,
        vision=owner.vision,
        rgb_tower=rgb_tower,
        depth_tower=depth_tower,
        rgbd_former=owner.rgbd._rpu_former,
        navdp=owner.navdp,
        rpu_execution=execution,
    )

    output_ids = output_ids.to(torch.long)
    image_idx = image_idx.to(torch.bool)
    position_ids = position_ids.to(torch.int64)

    def infer():
        with owner._internvla_execution_controller.execute():
            vision = owner.vision.forward(
                fixture["pixel_values"].float(), grid_thw
            ).float().cpu()
            text = torch.nn.functional.embedding(output_ids, embed_weight).clone()
            text[image_idx] = vision.reshape(-1, 3584).to(text.dtype)
            embeds = torch.cat([text, latent_queries], dim=1)
            vlm = owner.backbone.forward_embeds(
                embeds, position_ids
            ).float().cpu()[:, -4:, :]
            goal = owner.goal_glue(vlm).float()
            rgbd = owner.rgbd(
                fixture["rgb"].bfloat16(), fixture["depth"].bfloat16()
            ).float()
            return owner.navdp.predict_action(
                goal,
                rgbd,
                sample_num=sample_num,
                seed=inputs.get("seed", 1),
                init_noise=fixture["init_noise"].float(),
            ).float().cpu()

    return infer


prepare = _internvla_n1
