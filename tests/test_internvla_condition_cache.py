"""Condition identity shortcuts must preserve mutation freshness in both modes."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.internvla_n1.nextdit import NextDiTRPURuntime


@pytest.mark.parametrize("inference", [False, True])
def test_condition_mutation_refreshes_cached_binding(inference):
    owner = object.__new__(NextDiTRPURuntime)
    owner._condition_batch_source = None
    owner._condition_batch_version = -1
    owner._condition_batch_bindings = []
    owner._conditions = []
    calls = []

    def condition_state(condition):
        state = SimpleNamespace(source=condition.clone())
        calls.append(state)
        return 0, state

    owner._condition_state = condition_state
    with torch.inference_mode(inference):
        condition = torch.ones(1, 2, 3)
        first = owner._condition_bindings(condition)
        again = owner._condition_bindings(condition)
        assert torch.equal(again[0][1].source, first[0][1].source)
        condition.add_(1)
        changed = owner._condition_bindings(condition)
        assert torch.equal(changed[0][1].source, condition)
        assert not torch.equal(first[0][1].source, condition)
    assert len(calls) == (3 if inference else 2)
