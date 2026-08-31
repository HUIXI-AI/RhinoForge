from __future__ import annotations

from types import SimpleNamespace

import pytest

from rpu_backend.adapters.dinov3 import _dinov3_layers
from rpu_backend.api.errors import RPUBackendError


def test_dinov3_layers_accepts_direct_and_wrapped_hf_layouts_without_eager_fallback() -> None:
    direct = SimpleNamespace(layer=["direct"])
    assert _dinov3_layers(direct) == ["direct"]

    class Wrapped:
        model = SimpleNamespace(layer=["wrapped"])

        @property
        def layer(self):
            raise AssertionError("direct fallback must not be evaluated eagerly")

    assert _dinov3_layers(Wrapped()) == ["wrapped"]

    # A diagnostic/self-referential compatibility alias must still resolve the
    # real direct layer rather than recursing through ``model.model``.
    self_alias = SimpleNamespace(layer=["direct"])
    self_alias.model = self_alias
    assert _dinov3_layers(self_alias) == ["direct"]


def test_dinov3_layers_rejects_an_unknown_layout() -> None:
    with pytest.raises(RPUBackendError, match="encoder layers"):
        _dinov3_layers(SimpleNamespace())
