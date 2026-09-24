"""CPU regressions for atomic HALO action-prefix publication."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.halo import runtime
from rpu_backend.api.cache import RPUCache


@pytest.fixture
def prefix_runner(monkeypatch):
    from rpu_backend.api import causal_lm

    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    cfg = replace(
        runtime.HaloConfig(),
        num_layers=2,
        num_kv_heads=1,
        head_dim=16,
        prefix_len=16,
        attn_tp=1,
    )
    cache = RPUCache(2, 1, 48, 1, 16, device="cpu", attn_tp=1)
    runner = object.__new__(runtime.HaloStepRunner)
    runner.cfg = cfg
    runner.cache = cache
    runner._prefix_epoch = 0
    runner._closed = False

    state = SimpleNamespace(upload_count=0, fail_upload=None)
    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        device = args[0] if args else kwargs.get("device")
        if str(device) == "rpu":
            state.upload_count += 1
            if state.upload_count == state.fail_upload:
                raise RuntimeError("late prefix upload failed")
            if args:
                args = ("cpu", *args[1:])
            else:
                kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    return SimpleNamespace(runner=runner, state=state)


def _prefix(value):
    return [torch.full((16, 1, 16), value + layer) for layer in range(2)]


def _same_owners(actual, expected):
    return all(current is previous for current, previous in zip(actual, expected))


def test_successful_reinsert_publishes_one_complete_bank_and_advances_epoch(prefix_runner):
    runner, state = prefix_runner.runner, prefix_runner.state
    runner.insert_prefix(_prefix(1.0), _prefix(3.0))
    old_k, old_v = tuple(runner.cache.k_caches), tuple(runner.cache.v_caches)
    runner.cache.reset_to_position(19)

    runner.insert_prefix(_prefix(5.0), _prefix(7.0))

    assert state.upload_count == 8
    assert not any(current is previous
                   for current, previous in zip(runner.cache.k_caches, old_k))
    assert not any(current is previous
                   for current, previous in zip(runner.cache.v_caches, old_v))
    assert runner.cache.position == runner.cache._seen_tokens == 16
    assert runner._prefix_epoch == 2


def test_late_layer_shape_failure_preserves_complete_old_bank(prefix_runner):
    runner, state = prefix_runner.runner, prefix_runner.state
    runner.insert_prefix(_prefix(1.0), _prefix(3.0))
    old_k, old_v = tuple(runner.cache.k_caches), tuple(runner.cache.v_caches)
    runner.cache.reset_to_position(19)
    old_epoch = runner._prefix_epoch
    state.upload_count = 0
    bad_values = _prefix(7.0)
    bad_values[1] = torch.zeros(15, 1, 16)

    with pytest.raises(ValueError, match="layer 1 prefix shape"):
        runner.insert_prefix(_prefix(5.0), bad_values)

    assert state.upload_count == 0
    assert _same_owners(runner.cache.k_caches, old_k)
    assert _same_owners(runner.cache.v_caches, old_v)
    assert runner.cache.position == runner.cache._seen_tokens == 19
    assert runner._prefix_epoch == old_epoch


def test_late_layer_upload_failure_preserves_complete_old_bank(prefix_runner):
    runner, state = prefix_runner.runner, prefix_runner.state
    runner.insert_prefix(_prefix(1.0), _prefix(3.0))
    old_k, old_v = tuple(runner.cache.k_caches), tuple(runner.cache.v_caches)
    runner.cache.reset_to_position(19)
    old_epoch = runner._prefix_epoch
    state.upload_count = 0
    state.fail_upload = 4

    with pytest.raises(RuntimeError, match="late prefix upload failed"):
        runner.insert_prefix(_prefix(5.0), _prefix(7.0))

    assert state.upload_count == 4
    assert _same_owners(runner.cache.k_caches, old_k)
    assert _same_owners(runner.cache.v_caches, old_v)
    assert runner.cache.position == runner.cache._seen_tokens == 19
    assert runner._prefix_epoch == old_epoch
