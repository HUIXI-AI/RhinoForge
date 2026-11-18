"""Real HF visual-position callers across the supported Transformers versions."""
import importlib

import pytest
import torch

from rpu_backend.runtime.vision_position import canonical_vision_type, interpolate_vision_positions
from rpu_backend.adapters.qwen3_vl.vision import _compute_vision_position_idx_cpu
from rpu_backend.quant.load import _materialize_imagetext_meta_buffers


@pytest.mark.parametrize("family,hidden", [("qwen3_vl", 1024), ("qwen3_5", 1024),
                                            ("qwen3_5_moe", 1152), ("rhinovla", 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_position_precompute_matches_real_hf_forward(family, hidden, dtype):
    names = {"qwen3_vl": "Qwen3VL", "qwen3_5": "Qwen3_5", "qwen3_5_moe": "Qwen3_5Moe"}
    interpolate = interpolate_vision_positions
    if family == "rhinovla":
        from rpu_backend.adapters.rhinovla.vision import interpolate_vision_positions as interpolate
        family = "qwen3_vl"
    prefix = names[family]
    module = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    config_cls = getattr(module, prefix + "VisionConfig")
    model_cls = getattr(module, prefix + "VisionModel")
    config = config_cls(hidden_size=hidden, intermediate_size=4304 if hidden == 1152 else 4096,
        depth=1, num_heads=16, out_hidden_size=2048, deepstack_visual_indexes=[])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = model_cls(config).eval()
    # Execute the real forward's entire position path, without allocating block weights.
    model.patch_embed = torch.nn.Identity()
    model.merger = torch.nn.Identity()
    model.pos_embed = torch.nn.Embedding(2304, hidden, dtype=dtype)
    with torch.no_grad():
        values = torch.arange(2304 * hidden, dtype=torch.float32).remainder(97) / 97
        model.pos_embed.weight.copy_(values.reshape(2304, hidden))
    rotary = model.rotary_pos_emb
    if hasattr(rotary, "compute_axial_rope_parameters"):
        model.rotary_pos_emb = type(rotary)(config)
    else:
        model.rotary_pos_emb = type(rotary)(rotary.dim, rotary.theta)
    captured = {}

    class Capture(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            captured.update(kwargs)
            return hidden_states

    model.blocks = torch.nn.ModuleList([Capture()])
    grid = torch.tensor([[1, 4, 6], [2, 6, 4], [1, 8, 8]])
    count = int(grid.prod(-1).sum())
    inputs = torch.zeros(count, hidden, dtype=dtype)
    with torch.inference_mode():
        result = model(inputs, grid)
        positions = interpolate(model, grid)
    assert positions.dtype == dtype and positions.device.type == "cpu"
    torch.testing.assert_close(result.last_hidden_state, positions, rtol=0, atol=0)
    assert torch.isfinite(positions).all()
    indices = _compute_vision_position_idx_cpu(grid, 2).long()
    expected_indices = torch.cat([
        torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), -1)
        .reshape(h // 2, 2, w // 2, 2, 2).permute(0, 2, 1, 3, 4).reshape(-1, 2).repeat(t, 1)
        for t, h, w in grid.tolist()
    ])
    assert torch.equal(indices, expected_indices)
    frequencies = indices[:, :, None].float() * model.rotary_pos_emb.inv_freq.float()
    phases = frequencies.flatten(1).repeat(1, 2)
    cos, sin = captured["position_embeddings"]
    torch.testing.assert_close(cos, phases.cos(), rtol=0, atol=0)
    torch.testing.assert_close(sin, phases.sin(), rtol=0, atol=0)
    assert torch.equal(captured["cu_seqlens"], torch.tensor([0, 24, 48, 72, 136], dtype=torch.int32))


@pytest.mark.parametrize("family", ["qwen3_vl", "qwen3_5"])
def test_real_vl_meta_loader_restores_both_versions_rotary_buffers(family):
    if family == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration as model_cls
        from test_qwen3vl_core_config import _config
        restore = _materialize_imagetext_meta_buffers
    else:
        from transformers import Qwen3_5ForConditionalGeneration as model_cls
        from test_qwen35_core_profiles import _config
        from rpu_backend.adapters.qwen3_5.loader import _materialize_meta_buffers as restore
    with torch.device("meta"):
        model = model_cls(_config(2))
    restore(model)
    assert all(value.is_meta for value in model.parameters())
    assert all(value.device.type == "cpu" and torch.isfinite(value).all() for value in model.buffers())
    reference = 1 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
    torch.testing.assert_close(model.model.visual.rotary_pos_emb.inv_freq, reference, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["qwen3_vl", "qwen3_vl_vision"])
@pytest.mark.parametrize("fault", ["theta", "type", "partial", "head_dim"])
def test_alias_does_not_admit_nonstandard_rotary_semantics(name, fault):
    from test_qwen3vl_core_config import _config
    from rpu_backend.adapters.qwen3_vl import _qwen3_vl_core_topology, UnsupportedModelError

    config = _config(2)
    object.__setattr__(config.vision_config, "model_type", name)
    rope = {"rope_type": "axial", "rope_theta": 10000.0}
    if fault == "theta":
        rope["rope_theta"] = 20000.0
    elif fault == "type":
        rope["rope_type"] = "default"
    elif fault == "partial":
        rope["partial_rotary_factor"] = 0.5
    else:
        object.__setattr__(config.vision_config, "head_dim", 128)
    object.__setattr__(config.vision_config, "rope_parameters", rope)
    assert canonical_vision_type(config.vision_config, "qwen3_vl") is None
    with pytest.raises(UnsupportedModelError):
        _qwen3_vl_core_topology(config, {"model": {"num_cores": 6}})
