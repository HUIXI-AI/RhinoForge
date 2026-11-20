"""Board-free checks for the LightNav W8 calibration collector."""

from pathlib import Path
import argparse
import importlib.util
from types import SimpleNamespace

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "examples/lightnav_collect_activation_stats.py"
SPEC = importlib.util.spec_from_file_location("lightnav_collect_activation_stats", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


class Projection(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.in_features = width

    def forward(self, value):
        return value


def test_collector_emits_every_projection_as_positive_float32():
    layers = []
    for _ in range(36):
        layers.append(SimpleNamespace(
            self_attn=SimpleNamespace(
                q_proj=Projection(3), k_proj=Projection(3), v_proj=Projection(3),
                o_proj=Projection(4),
            ),
            mlp=SimpleNamespace(
                gate_proj=Projection(3), up_proj=Projection(3), down_proj=Projection(5),
            ),
        ))
    model = SimpleNamespace(model=SimpleNamespace(
        language_model=SimpleNamespace(layers=layers),
    ))
    moments = COLLECTOR.ActivationMoments()
    handles = moments.register(model)
    for layer in layers:
        for path in COLLECTOR.PROJECTIONS.values():
            module = layer
            for name in path.split("."):
                module = getattr(module, name)
            module(torch.arange(1, module.in_features + 1, dtype=torch.float16).repeat(2, 1))
    for handle in handles:
        handle.remove()

    result = moments.finish()
    assert len(result) == 252
    assert torch.equal(result["layers.0.q"], torch.tensor([1.0, 4.0, 9.0]))
    assert all(value.dtype == torch.float32 and value.device.type == "cpu" for value in result.values())


def test_teacher_tokens_must_match_requested_episode_frames(tmp_path):
    path = tmp_path / "teacher.json"
    path.write_text(
        '{"instruction":"follow","history_frames":9,"output_tokens":5,'
        '"inferences":[{"frame_index":9,"image":"9.jpg","token_ids":[1,2,3,4,5]}]}',
        encoding="utf-8",
    )
    images = [f"{i}.jpg" for i in range(10)]
    assert COLLECTOR.load_teacher_tokens(path, (9,), images, "follow", 9) == {
        9: [1, 2, 3, 4, 5],
    }
    with pytest.raises(ValueError, match="missing frame_index=8"):
        COLLECTOR.load_teacher_tokens(path, (8,), images, "follow", 9)
    with pytest.raises(ValueError, match="instruction/history_frames"):
        COLLECTOR.load_teacher_tokens(path, (9,), images, "turn left", 9)
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        COLLECTOR.parse_frame_indices("9,9")
