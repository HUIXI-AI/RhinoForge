"""Host-only real prefix helper reuse across request autograd contexts."""
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters.hy_vla.runtime import HyVlaRunner


@pytest.mark.parametrize("images", [1, 2])
def test_prefix_template_reuses_mutable_storage_and_snapshots_tokens(images):
    # Skip hardware-owning construction; exercise the actual CPU prefix methods.
    runner = object.__new__(HyVlaRunner)
    cfg = NS(grid=2, proj_dim=3, tok_bos=0, tok_user=1,
             tok_vision_start=2, tok_vision_split=3, tok_vision_end=4)
    tokens = torch.arange(48, dtype=torch.float32).reshape(16, 3)
    runner.w = NS(cfg=cfg, tok_emb=tokens)
    runner._prefix_template = True
    runner._prefix_key = runner._prefix_val = None
    lang = torch.tensor([5, 6])
    first_images = [torch.full((4, 3), float(i + 20)) for i in range(images)]
    with torch.inference_mode():
        first = runner.assemble_prefix(first_images, lang)
        first_snapshot = first.clone()
    second_images = [x + 10 for x in first_images]
    with torch.no_grad():
        second = runner.assemble_prefix(second_images, lang)
    assert second is first
    assert not second.is_inference()
    assert not torch.equal(second, first_snapshot)
    assert torch.equal(second[0, -2:], tokens[lang].half())
    assert all(torch.equal(slot, image.reshape(2, 2, 3).half())
               for slot, image in zip(runner._prefix_val[1], second_images))
    # Reusing and mutating the caller's input must miss the cloned cache key.
    lang[0] = 7
    with torch.no_grad():
        third = runner.assemble_prefix(second_images, lang)
    assert third is not second
    assert torch.equal(third[0, -2:], tokens[lang].half())
    assert torch.equal(second[0, -2:], tokens[torch.tensor([5, 6])].half())
    # Compare with the existing uncached assembly path, without new math.
    runner._prefix_template = False
    with torch.no_grad():
        uncached = runner.assemble_prefix(second_images, lang)
    assert torch.equal(third, uncached.half())
