"""Explicit Pi input noise is validated and forwarded without diagnostic state."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.api.policy import Pi05Policy


def make_policy():
    calls = []
    policy = object.__new__(Pi05Policy)
    policy._rpu_ready = True
    policy._optimized_profile = None
    policy._rpu_execution = None
    def record(batch, **kwargs):
        calls.append(kwargs)
        return kwargs.get("noise")
    policy._lerobot_policy = SimpleNamespace(
        config=SimpleNamespace(chunk_size=50, max_action_dim=32),
        predict_action_chunk=record)
    policy._adapter = SimpleNamespace(prepare_graphs=record)
    return policy, calls


@pytest.mark.parametrize("method", ["prepare_graphs", "predict_action_chunk"])
def test_explicit_noise_preserves_input_and_uses_normal_policy_call(method):
    policy, calls = make_policy()
    noise = torch.randn(1, 50, 32)
    original = noise.clone()
    batch = {"observation.state": torch.ones(1, 32)}
    result = getattr(policy, method)(batch, noise=noise, num_steps=10)
    assert result is noise and calls[0]["noise"] is noise
    assert torch.equal(noise, original)


@pytest.mark.parametrize("method", ["prepare_graphs", "predict_action_chunk"])
@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "state", "object"])
def test_bad_noise_fails_before_policy_or_graph_mutation(method, kind):
    policy, calls = make_policy()
    noise = torch.zeros(1, 50, 32)
    batch = {"observation.state": torch.ones(1, 32)}
    if kind == "shape": noise = noise[:, :, :7]
    elif kind == "dtype": noise = noise.half()
    elif kind == "nan": noise[0, 0, 0] = float("nan")
    elif kind == "state": batch = {}
    elif kind == "object": noise = {"noise": noise}
    with pytest.raises(ValueError, match="noise"):
        getattr(policy, method)(batch, noise=noise, num_steps=10)
    assert not calls
