"""Qwen3.5 Vision is requested by the bound execution config, not an opt-in.

The adapter and the lazy Vision install see the normalized, read-only
execution Mapping; a check that only accepts ``dict`` would miss the declared
``vision`` stage of every public image template.
"""
import json
from pathlib import Path
import tomllib

import pytest
from transformers import Qwen3_5Config

from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter
from rpu_backend.adapters.qwen3_5_moe import Qwen3_5MoeAdapter
from rpu_backend.adapters.qwen3_5.vision import qwen35_vision_requested
from rpu_backend.api._execution import normalize_rpu_execution
from rpu_backend.runtime import UnsupportedModelError

ROOT = Path(__file__).resolve().parents[1]
QWEN35 = ROOT / "examples/configs/qwen3_5"


def _normalize(stages):
    # Same normalization the adapter applies before storing _rpu_execution.
    return normalize_rpu_execution(stages, entry_point="Qwen3_5Adapter",
                                   supported=Qwen3_5Adapter._EXECUTION_SUPPORTED)


def _bound(path):
    doc = tomllib.loads(path.read_text())
    raw = doc.get("rpu_execution", {})
    stages = {key: value for key, value in raw.items() if key != "model"}
    if doc.get("model", {}).get("architecture") == "Qwen3_5MoeForConditionalGeneration":
        return normalize_rpu_execution(stages, entry_point="Qwen3_5MoeAdapter",
                                       supported=Qwen3_5MoeAdapter._EXECUTION_SUPPORTED)
    return _normalize(stages)


@pytest.fixture(autouse=True)
def _no_legacy_env(monkeypatch):
    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)


@pytest.mark.parametrize("path", sorted((QWEN35 / "vl").glob("[0-9]*b/*.toml")),
                         ids=lambda path: path.relative_to(QWEN35).as_posix())
def test_every_dense_image_template_requests_vision_when_bound(path):
    execution = _bound(path)
    assert not isinstance(execution, dict)
    assert qwen35_vision_requested(execution) is True


@pytest.mark.parametrize("path", sorted((QWEN35 / "text").glob("[0-9]*b/*.toml")),
                         ids=lambda path: path.relative_to(QWEN35).as_posix())
def test_text_templates_never_request_vision(path):
    assert qwen35_vision_requested(_bound(path)) is False


def test_legacy_env_still_requests_vision_without_an_execution_config(monkeypatch):
    assert qwen35_vision_requested(None) is False
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    assert qwen35_vision_requested(None) is True


def test_awq_text_only_recipe_rejects_a_bound_vision_stage_before_weights():
    config = Qwen3_5Config.from_dict(json.loads(
        (ROOT / "tests/fixtures/qwen35_quanttrio_awq/2b.json").read_text()))
    image = _normalize({"prefill": {"chunk_size": "auto"}, "vision": {"chunk_size": "auto"}})
    with pytest.raises(UnsupportedModelError, match="supports pure text only"):
        Qwen3_5Adapter.preflight_execution(config, image)
