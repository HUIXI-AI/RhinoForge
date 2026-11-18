#!/usr/bin/env python3
"""D-12 acceptance tests for Qwen3 profile certification guards.

D-12 + iter1 MEDIUM-10: `Qwen3Adapter.__init__(model)` checks the 4-tuple
profile `(hidden_size, intermediate_size, num_hidden_layers,
num_key_value_heads)` against `_SUPPORTED_PROFILES`. Models whose profile is
not in the registry raise `UnsupportedModelError` with a documented
remediation message.
"""
import sys

import pytest

from rpu_backend.api.errors import UnsupportedModelError
from rpu_backend.adapters.qwen3 import (
    Qwen3Adapter,
    _SUPPORTED_PROFILES,
    _W8A16_ONLY_PROFILES,
)


def _make_fake_qwen3_model(*, hidden_size: int, intermediate_size: int,
                            num_hidden_layers: int, num_key_value_heads: int = 8,
                            num_attention_heads: int = 32,
                            quant_config=None):
    """Construct a minimal stand-in for HF Qwen3ForCausalLM exposing the FULL
    4-tuple profile that the iter1+iter2 MEDIUM-10 envelope guard checks.

    All four dims are required because `Qwen3Adapter.__init__` reads
    `(hidden_size, intermediate_size, num_hidden_layers, num_key_value_heads)`
    and would raise AttributeError (not UnsupportedModelError) if any is missing.
    """
    class FakeConfig:
        pass
    cfg = FakeConfig()
    cfg.hidden_size = hidden_size
    cfg.intermediate_size = intermediate_size
    cfg.num_hidden_layers = num_hidden_layers
    cfg.num_key_value_heads = num_key_value_heads
    cfg.num_attention_heads = num_attention_heads
    cfg.architectures = ["Qwen3ForCausalLM"]
    if quant_config is not None:
        cfg.quant_config = quant_config

    class FakeModel:
        def __init__(self, config):
            self.config = config
            # Successful loader admission binds the existing outer/inner
            # execution Session, even before any device installation.
            self.model = FakeConfig()
            self.model.config = config
        def to(self, *a, **k):
            return self
        def eval(self):
            return self
        def named_modules(self):
            return iter(())
        def parameters(self):
            return iter(())
    return FakeModel(cfg)


# Real Qwen3 profiles. Large dense profiles are admitted by the reduced-core
# geometry; quantized recipes still have their own loader contracts.
_Q3_14B_PROFILE = dict(hidden_size=5120, intermediate_size=17408, num_hidden_layers=40, num_key_value_heads=8, num_attention_heads=40)
_Q3_32B_PROFILE = dict(hidden_size=5120, intermediate_size=25600, num_hidden_layers=64, num_key_value_heads=8, num_attention_heads=64)
_Q3_8B_PROFILE = dict(hidden_size=4096, intermediate_size=12288, num_hidden_layers=36, num_key_value_heads=8, num_attention_heads=32)


def test_qwen3_14b_fp16_has_a_reduced_core_profile():
    """14B FP16 reaches the same geometry gate as the smaller dense models."""
    model = _make_fake_qwen3_model(**_Q3_14B_PROFILE)
    Qwen3Adapter.preflight(model.config)


def test_qwen3_14b_fp16_via_from_pretrained_reaches_model_load():
    """The config gate no longer rejects the dense 14B profile before loading."""
    from unittest.mock import patch, MagicMock

    fake_14b = _make_fake_qwen3_model(**_Q3_14B_PROFILE)

    with patch("rpu_backend.api.causal_lm.AutoConfig") as MockAutoConfig, \
         patch("rpu_backend.api.causal_lm.AutoModelForCausalLM") as MockAutoModel:
        MockAutoConfig.from_pretrained.return_value = fake_14b.config
        MockAutoModel.from_pretrained.return_value = fake_14b
        from rpu_backend import RPUModelForCausalLM
        import torch
        model = RPUModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", dtype=torch.float16)
        assert model is fake_14b
        MockAutoConfig.from_pretrained.assert_called_once()
        MockAutoModel.from_pretrained.assert_called_once()


def test_qwen3_14b_w8a16_preflight_is_allowed():
    """The existing 14B W8A16 loader remains admitted."""
    model = _make_fake_qwen3_model(
        **_Q3_14B_PROFILE,
        quant_config={"method": "w8a16"},
    )
    Qwen3Adapter.preflight(model.config)


def test_qwen3_14b_w8a16_via_from_pretrained_skips_fp16_load():
    """W8A16 14B may pass preflight, but must still use the W8A16 loader path
    rather than AutoModelForCausalLM.from_pretrained."""
    from unittest.mock import patch

    fake_14b = _make_fake_qwen3_model(
        **_Q3_14B_PROFILE,
        quant_config={"method": "w8a16"},
    )

    with patch("rpu_backend.api.causal_lm.AutoConfig") as MockAutoConfig, \
         patch("rpu_backend.api.causal_lm.AutoModelForCausalLM") as MockAutoModel, \
         patch("rpu_backend.quant.load.load_w8a16_model") as MockW8A16Loader:
        MockAutoConfig.from_pretrained.return_value = fake_14b.config
        MockW8A16Loader.return_value = fake_14b
        from rpu_backend import RPUModelForCausalLM
        import torch

        model = RPUModelForCausalLM.from_pretrained(
            "local/qwen3-14b-w8a16-lmhead-int8",
            dtype=torch.float16,
        )

        assert model is fake_14b
        MockAutoConfig.from_pretrained.assert_called_once()
        MockAutoModel.from_pretrained.assert_not_called()
        MockW8A16Loader.assert_called_once()


def test_qwen3_8b_invalid_exact_chunk_fails_before_weight_load():
    """A 16-aligned value is not necessarily legal for this SDPA geometry.

    The stale 8B/c224 benchmark pin must be rejected from config alone, before
    the multi-GB checkpoint loader is called.
    """
    from unittest.mock import patch

    fake_8b = _make_fake_qwen3_model(**_Q3_8B_PROFILE)
    with patch("rpu_backend.api.causal_lm.AutoConfig") as MockAutoConfig, \
         patch("rpu_backend.api.causal_lm.AutoModelForCausalLM") as MockAutoModel:
        MockAutoConfig.from_pretrained.return_value = fake_8b.config
        from rpu_backend import RPUModelForCausalLM
        import torch

        with pytest.raises(UnsupportedModelError, match="chunk ceiling 128"):
            RPUModelForCausalLM.from_pretrained(
                "local/qwen3-8b",
                dtype=torch.float16,
                rpu_execution={"prefill": {"chunk_size": 224}},
            )

        MockAutoConfig.from_pretrained.assert_called_once()
        MockAutoModel.from_pretrained.assert_not_called()


def test_qwen3_32b_fp16_exceeds_memory_capacity():
    model = _make_fake_qwen3_model(**_Q3_32B_PROFILE)
    with pytest.raises(UnsupportedModelError, match="device memory capacity"):
        Qwen3Adapter.preflight(model.config)


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_qwen3_32b_fp16_rejects_before_weight_load(cores):
    from unittest.mock import patch
    import torch
    from rpu_backend import RPUModelForCausalLM

    model = _make_fake_qwen3_model(**_Q3_32B_PROFILE)
    with patch("rpu_backend.api.causal_lm.AutoConfig") as config_loader, \
         patch("rpu_backend.api.causal_lm.AutoModelForCausalLM") as weights_loader:
        config_loader.from_pretrained.return_value = model.config
        with pytest.raises(UnsupportedModelError, match="W8A16.*W4A16"):
            RPUModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-32B", dtype=torch.float16,
                rpu_execution={"model": {"num_cores": cores}},
            )
        weights_loader.from_pretrained.assert_not_called()


def test_supported_envelope_unchanged():
    """Sanity: certified FP16 and W8A16 profile sets remain distinct."""
    assert (1024,  3072, 28, 8) in _SUPPORTED_PROFILES, "0.6B profile must be supported"
    assert (2048,  6144, 28, 8) in _SUPPORTED_PROFILES, "1.7B profile must be supported"
    assert (2560,  9728, 36, 8) in _SUPPORTED_PROFILES, "4B profile must be supported"
    assert (5120, 17408, 40, 8) in _SUPPORTED_PROFILES, "14B dense profile must be supported"
    assert (5120, 25600, 64, 8) not in _SUPPORTED_PROFILES, "32B requires quantized loading"
    q3_8b_profile = (4096, 12288, 36, 8)
    assert q3_8b_profile in _SUPPORTED_PROFILES, "8B FP16 must remain certified"


def main() -> int:
    return pytest.main([__file__])


if __name__ == "__main__":
    sys.exit(main())
