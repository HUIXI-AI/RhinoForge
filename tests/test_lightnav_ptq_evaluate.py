"""Board-free checks for the customer LightNav PTQ evaluator."""

from pathlib import Path
import importlib.util
import json

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "examples/lightnav_ptq_evaluate.py"
SPEC = importlib.util.spec_from_file_location("lightnav_ptq_evaluate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


def test_load_episode_and_parse_reference_tokens(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    for index in range(1, 3):
        (episode / f"image_{index:06d}.jpg").write_bytes(b"jpeg")
    rows = [
        {"step": index, "instruction": "follow", "raw_text": raw}
        for index, raw in enumerate((
            "<apos_1><opos_2><act_l0_3><act_l1_4><act_l2_5><|im_end|>",
            "<apos_6><opos_7><act_l0_8><act_l1_9><act_l2_10><|im_end|>",
        ), 1)
    ]
    (episode / "action.json").write_text(json.dumps(rows), encoding="utf-8")
    instruction, images, loaded = EVALUATE.load_ptq_episode(episode)
    vocabulary = {token: index for index, token in enumerate(
        EVALUATE.TOKEN_PATTERN.findall(rows[0]["raw_text"])
    )}
    assert instruction == "follow"
    assert [image.name for image in images] == ["image_000001.jpg", "image_000002.jpg"]
    assert EVALUATE.parse_action_tokens(loaded[0]["raw_text"], vocabulary) == [0, 1, 2, 3, 4]


def test_reference_and_timing_validation_fail_closed():
    with pytest.raises(ValueError, match="five LightNav action tokens"):
        EVALUATE.parse_action_tokens("<apos_1><opos_2>", {})
    summary = EVALUATE.summarize_timings([
        {key: value for key in EVALUATE.TIMING_KEYS} for value in (200.0, 300.0)
    ])
    assert summary["frame_total_mean_ms"] == 250.0
    assert summary["frame_total_p95_ms"] == 300.0
    assert summary["mean_fps"] == 4.0


def test_episode_rejects_non_object_actions(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "action.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="entries must be objects"):
        EVALUATE.load_ptq_episode(episode)


def test_load_episode_accepts_one_contiguous_png_sequence(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "image_000001.png").write_bytes(b"png")
    rows = [{"step": 1, "instruction": "follow", "raw_text": "unused"}]
    (episode / "action.json").write_text(json.dumps(rows), encoding="utf-8")

    _, images, _ = EVALUATE.load_ptq_episode(episode)

    assert [image.name for image in images] == ["image_000001.png"]


def test_load_episode_rejects_mixed_jpeg_and_png_sequences(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "image_000001.jpg").write_bytes(b"jpeg")
    (episode / "image_000001.png").write_bytes(b"png")
    rows = [{"step": 1, "instruction": "follow", "raw_text": "unused"}]
    (episode / "action.json").write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one image format"):
        EVALUATE.load_ptq_episode(episode)


def test_find_episode_paths_ignores_metadata_directories(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "action.json").write_text("[]", encoding="utf-8")
    (tmp_path / "logs").mkdir()

    assert EVALUATE.find_episode_paths(tmp_path) == [episode]
