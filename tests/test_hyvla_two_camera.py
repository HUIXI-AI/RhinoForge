"""Two-camera input geometry through preprocessing, public validation and composition."""
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters.hy_vla import preprocess
from rpu_backend.adapters.hy_vla.runtime import HyVlaRunner
from rpu_backend.api.hy_embodied import HyEmbodiedPolicy, _validate_runner_observation


@pytest.mark.parametrize("text", [1, 16, 17, 32, 48, 64])
def test_two_camera_padding_is_invisible_and_keeps_real_suffix_positions(text):
    class Tokenizer:
        def __call__(self, *_args, max_length, **_kwargs):
            mask = torch.arange(max_length)[None] < text
            return {"input_ids": mask.long() * 5, "attention_mask": mask}

    obs = preprocess.build_observation(
        [torch.full((1, 3, 8, 8), .25)] * 2, "fixed instruction", Tokenizer(),
        state=torch.zeros(1, 32), noise_seed=3)
    physical = ((118 + text + 15) // 16) * 16
    _validate_runner_observation(obs, n_cameras=2, prefix_len=physical)
    assert obs["n_valid"] == 118 + text
    assert obs["lang_tokens"].shape == (64,)
    assert obs["prefill_mask"].shape == (192, 192)
    assert obs["denoise_mask"].shape == (51, 243)
    assert obs["modality_mask"].sum() == 2 * 49
    assert not obs["modality_mask"][118:].any()
    assert not obs["prefill_mask"][obs["n_valid"]:].any()
    assert not obs["prefill_mask"][:, obs["n_valid"]:].any()
    assert not obs["denoise_mask"][:, obs["n_valid"]:192].any()
    assert torch.equal(obs["suffix_pos"], torch.arange(51) + obs["n_valid"])
    assert torch.equal(obs["prefill_pos"][182:], torch.full((10,), obs["n_valid"] - 1))
    with pytest.raises(ValueError, match="有效 prefix"):
        _validate_runner_observation(obs, n_cameras=2, prefix_len=physical-16)
    with pytest.raises(ValueError, match="3 张图"):
        _validate_runner_observation(obs, n_cameras=3, prefix_len=physical)


def test_padded_assembly_matches_original_rows_and_retains_mutable_cache():
    runner = object.__new__(HyVlaRunner)
    runner.prefix_len = 192
    cfg = NS(grid=7, proj_dim=3, tok_bos=0, tok_user=1,
             tok_vision_start=2, tok_vision_split=3, tok_vision_end=4)
    runner.w = NS(cfg=cfg, tok_emb=torch.arange(300).reshape(100, 3).float())
    runner._prefix_key = runner._prefix_val = None
    lang = torch.arange(5, 69)
    images = [torch.full((49, 3), float(i + 100)) for i in range(2)]
    runner._prefix_template = False
    expected = runner.assemble_prefix(images, lang).half()
    assert expected.shape == (1, 192, 3)
    assert torch.equal(expected[0, 118:182], runner.w.tok_emb[lang].half())
    assert not expected[:, 182:].count_nonzero()
    runner._prefix_template = True
    with torch.inference_mode():
        actual = runner.assemble_prefix(images, lang)
    assert torch.equal(actual, expected)
    assert not actual.is_inference()
    with torch.no_grad():
        changed = runner.assemble_prefix([image + 1 for image in images], lang)
    assert changed is actual
    assert not changed[:, 182:].count_nonzero()
    assert torch.equal(changed[0, 118:182], expected[0, 118:182])


def test_public_camera_order_and_count_boundary(tmp_path):
    for cameras in (2, 3):
        policy = HyEmbodiedPolicy.from_checkpoint(tmp_path, n_cameras=cameras, prefix_len=192)
        assert policy.camera_names == list(preprocess.CAMERA_ORDER[:cameras])
        default = HyEmbodiedPolicy.from_checkpoint(tmp_path, n_cameras=cameras)
        assert default._prefix_len == (192 if cameras == 2 else 240)
    with pytest.raises(ValueError, match="prefix_len"):
        HyEmbodiedPolicy.from_checkpoint(tmp_path, n_cameras=2, prefix_len=208)
    for cameras in (0, 4, True, 2.0):
        with pytest.raises(ValueError, match="2 或 3 相机"):
            HyEmbodiedPolicy.from_checkpoint(tmp_path, n_cameras=cameras)


@pytest.mark.parametrize("text", [1, 32, 64])
def test_one_camera_pads_to_the_action_expert_floor(text, tmp_path):
    class Tokenizer:
        def __call__(self, *_args, max_length, **_kwargs):
            mask = torch.arange(max_length)[None] < text
            return {"input_ids": mask.long() * 5, "attention_mask": mask}

    obs = preprocess.build_observation(
        [torch.full((1, 3, 8, 8), .25)], "fixed instruction", Tokenizer(),
        state=torch.zeros(1, 32), noise_seed=3)
    _validate_runner_observation(obs, n_cameras=1, prefix_len=128)
    assert obs["n_valid"] == 60 + text
    assert obs["prefill_mask"].shape == (128, 128)
    assert obs["denoise_mask"].shape == (51, 179)
    assert obs["modality_mask"].sum() == 49
    assert not obs["prefill_mask"][obs["n_valid"]:].any()
    assert not obs["denoise_mask"][:, obs["n_valid"]:128].any()
    assert torch.equal(obs["suffix_pos"], torch.arange(51) + obs["n_valid"])
    policy = HyEmbodiedPolicy.from_checkpoint(tmp_path, n_cameras=1)
    assert policy._prefix_len == 128 and policy.camera_names == list(preprocess.CAMERA_ORDER[:1])
    with pytest.raises(ValueError, match="prefix_len"):
        HyEmbodiedPolicy.from_checkpoint(tmp_path, n_cameras=1, prefix_len=112)


@pytest.mark.parametrize("cameras, prefix, expected", [(1, "auto", None), (2, None, None), (3, 240, 240), (4, "auto", None)])
def test_public_runner_installs_the_image_count(monkeypatch, tmp_path, cameras, prefix, expected):
    from PIL import Image
    import rpu_backend.api as api
    from examples.runners.hy_vla import runner as hy_runner

    seen = {}

    class Policy:
        @classmethod
        def from_checkpoint(cls, path, **kwargs):
            seen.update(kwargs)
            return cls()

        def to(self, device):
            return self

    monkeypatch.setattr(api, "HyEmbodiedPolicy", Policy, raising=False)
    monkeypatch.setattr(hy_runner, "_checkpoint", lambda config: tmp_path)
    monkeypatch.setattr(hy_runner, "policy_dtype", lambda example: "fp16")
    paths = []
    for index in range(cameras):
        Image.new("RGB", (6, 4), (index, 0, 0)).save(tmp_path / f"{index}.png")
        paths.append(str(tmp_path / f"{index}.png"))
    inputs = {"images": paths, "instruction": "move"}
    if prefix is not None:
        inputs["prefix_len"] = prefix
    runner = NS(source_config={"example": {}, "input": inputs})
    if cameras == 4:
        with pytest.raises(ValueError, match="one to three camera images"):
            hy_runner.prepare(runner)
        assert not seen
        return
    hy_runner.prepare(runner)
    # auto/omitted leaves the bucket to the policy: ceil16(2 + 58C + 64) for C cameras.
    assert seen["n_cameras"] == cameras and seen["prefix_len"] == expected
