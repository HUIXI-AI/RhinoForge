"""Lazy Vision cold tensors remain mutable when later requests use no_grad."""
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters.qwen3_5 import vision


@pytest.mark.parametrize("lazy", [False, True])
def test_vision_cold_install_does_not_create_inference_buffers(monkeypatch, lazy):
    tower = NS(config=NS(), blocks=[], merger=object())
    model = NS(model=NS(visual=tower))
    cold_modes = []

    def allocate_cold_buffers(*args, **kwargs):
        # CPU doubles for the native position-index backing and the dummy KV
        # buffers allocated by the cold implementation. Forward updates these
        # owners in place; their context must not inherit the first request.
        cold_modes.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        tower._rpu_vision_position_idx_keepalive = torch.zeros(8, 2, dtype=torch.int16)
        tower.kv = torch.zeros(8, dtype=torch.float16)
        return 7

    monkeypatch.setattr(vision, "_install_qwen3_5_vision_for_rpu_impl", allocate_cold_buffers)
    with torch.inference_mode():
        if lazy:
            handle = vision._install_lazy_qwen3_5_vision(model, NS(execution_generation=3))
        else:
            handle = vision.install_qwen3_5_vision_for_rpu(tower)
        assert torch.is_inference_mode_enabled()
    assert handle == 7
    # Match the next request's position refresh and cache reset under no_grad.
    with torch.no_grad():
        tower._rpu_vision_position_idx_keepalive.narrow(0, 0, 4).copy_(
            torch.ones(4, 2, dtype=torch.int16))
        tower.kv.zero_()
    assert cold_modes == [(False, False)]
    assert not tower._rpu_vision_position_idx_keepalive.is_inference()
    assert tower._rpu_vision_position_idx_keepalive._version == 1
    assert tower.kv._version == 1


def test_failed_vision_cold_install_restores_request_context(monkeypatch):
    tower = NS(config=NS(), blocks=[], merger=object())
    modes = []

    def fail(*args, **kwargs):
        modes.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        raise ValueError("cold failure")

    monkeypatch.setattr(vision, "_install_qwen3_5_vision_for_rpu_impl", fail)
    with torch.inference_mode():
        with pytest.raises(ValueError, match="cold failure"):
            vision.install_qwen3_5_vision_for_rpu(tower)
        assert torch.is_inference_mode_enabled()
    assert modes == [(False, False)]
