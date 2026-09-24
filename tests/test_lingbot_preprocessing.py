from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rpu_backend.api.lingbot2 import Lingbot2Policy
from rpu_backend.api.errors import RPUBackendError


class _Tokenizer:
    def __init__(self) -> None:
        self.chat_inputs = []
        self.tokenize_inputs = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_inputs.append((messages, kwargs))
        return "<chat>public prompt</chat>"

    def __call__(self, prompt, **kwargs):
        self.tokenize_inputs.append((prompt, kwargs))
        return {
            "input_ids": torch.arange(72).reshape(1, 72),
            "attention_mask": torch.ones(1, 72),
        }


@pytest.mark.parametrize(
    ("enabled", "expected_prompt", "chat_calls"),
    ((True, "<chat>public prompt</chat>", 1), (False, "pick", 0)),
)
def test_lingbot_instruction_follows_public_chat_template_config(
    enabled: bool, expected_prompt: str, chat_calls: int
) -> None:
    tokenizer = _Tokenizer()
    policy = Lingbot2Policy.__new__(Lingbot2Policy)
    policy._processor = SimpleNamespace(tokenizer=tokenizer)
    policy._max_lang_tokens = 72
    policy._use_qwen3_chat_template = enabled

    tokens, mask = policy._encode_instruction("pick")

    assert tokens.shape == mask.shape == (1, 72)
    assert len(tokenizer.chat_inputs) == chat_calls
    assert tokenizer.tokenize_inputs == [
        (
            expected_prompt,
            {
                "return_tensors": "pt",
                "padding": "max_length",
                "padding_side": "right",
                "truncation": True,
                "max_length": 72,
            },
        )
    ]


def test_lingbot_training_config_resolves_prompt_format_before_load(tmp_path) -> None:
    assert Lingbot2Policy.from_checkpoint(tmp_path)._use_qwen3_chat_template is True
    config = tmp_path / "lingbotvla_cli.yaml"
    config.write_text("model:\n  use_qwen3_chat_template: false\n", encoding="utf-8")
    assert Lingbot2Policy.from_checkpoint(tmp_path)._use_qwen3_chat_template is False
    config.write_text("model:\n  use_qwen3_chat_template: raw\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be true or false"):
        Lingbot2Policy.from_checkpoint(tmp_path)


def test_lingbot_public_processor_fails_before_runtime_mutation(
    tmp_path, monkeypatch
) -> None:
    built = []
    adapter = types.ModuleType("rpu_backend.adapters.lingbot_vla_v2")
    adapter.build_lingbot_vla_v2 = lambda *args, **kwargs: built.append(True)
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)

    missing = Lingbot2Policy.from_checkpoint(
        tmp_path,
        runtime_env={"RPU_LINGBOT2_ALLOW_UNVALIDATED": "1"},
    )
    with pytest.raises(RPUBackendError, match="requires qwen3vl_base"):
        missing.to("rpu")
    assert missing._rpu_build_started is False
    assert built == []

    qwen_base = tmp_path / "qwen"
    no_template = Lingbot2Policy.from_checkpoint(
        tmp_path,
        qwen3vl_base=qwen_base,
        runtime_env={"RPU_LINGBOT2_ALLOW_UNVALIDATED": "1"},
    )
    no_template._processor = SimpleNamespace(
        tokenizer=lambda *_args, **_kwargs: {},
        image_processor=lambda *_args, **_kwargs: {},
    )
    with pytest.raises(RPUBackendError, match="required public Qwen3-VL processor"):
        no_template.to("rpu")
    assert no_template._rpu_build_started is False
    assert built == []


def test_lingbot_raw_prompt_keeps_checkpoint_processor_compatibility(tmp_path) -> None:
    config = tmp_path / "lingbotvla_cli.yaml"
    config.write_text("model:\n  use_qwen3_chat_template: false\n", encoding="utf-8")
    policy = Lingbot2Policy.from_checkpoint(tmp_path)
    policy._processor = object()

    policy._preflight_processor_contract()

    assert policy._processor_source == str(tmp_path)
    assert policy._processor is not None


class _ImageProcessor:
    def __init__(self) -> None:
        self.inputs = []

    def __call__(self, image):
        self.inputs.append(image.clone())
        height, width = image.shape[-2:]
        pixels = image.permute(1, 2, 0).reshape(1, height * width, 3)
        return {
            "pixel_values": pixels,
            "image_grid_thw": torch.tensor([[1, height, width]]),
        }


def test_lingbot_exact_images_match_public_per_camera_float_pipeline() -> None:
    from torchvision.transforms.v2 import Resize

    frames = [
        np.array(
            [
                [[0 + offset, 10 + offset, 20 + offset], [30 + offset, 40 + offset, 50 + offset]],
                [[60 + offset, 70 + offset, 80 + offset], [90 + offset, 100 + offset, 110 + offset]],
            ],
            dtype=np.uint8,
        )
        for offset in (0, 20, 40)
    ]
    image_processor = _ImageProcessor()
    policy = Lingbot2Policy.__new__(Lingbot2Policy)
    policy._processor = SimpleNamespace(image_processor=image_processor)
    policy._n_cameras = 3
    policy._image_size = 3
    policy._preproc_mode = "exact"

    pixels, grids = policy._encode_images(frames)

    resize = Resize((3, 3), antialias=True)
    expected_inputs = [
        resize(torch.as_tensor(frame).permute(2, 0, 1).contiguous().float())
        for frame in frames
    ]
    assert len(image_processor.inputs) == 3
    assert all(torch.equal(actual, expected) for actual, expected in zip(image_processor.inputs, expected_inputs))
    expected_pixels = torch.cat(
        [value.permute(1, 2, 0).reshape(1, 9, 3) for value in expected_inputs]
    )
    assert torch.equal(pixels, expected_pixels)
    assert grids.tolist() == [[1, 3, 3], [1, 3, 3], [1, 3, 3]]
