import sys
import pytest
from rpu_backend.api.errors import UnsupportedModelError
from rpu_backend.adapters.qwen3 import Qwen3Adapter, _SUPPORTED_PROFILES, _W8A16_ONLY_PROFILES

def _make_fake_qwen3_model(*, hidden_size: int, intermediate_size: int, num_hidden_layers: int, num_key_value_heads: int=8, num_attention_heads: int=32, quant_config=None):

    class FakeConfig:
        pass
    cfg = FakeConfig()
    cfg.hidden_size = hidden_size
    cfg.intermediate_size = intermediate_size
    cfg.num_hidden_layers = num_hidden_layers
    cfg.num_key_value_heads = num_key_value_heads
    cfg.num_attention_heads = num_attention_heads
    cfg.architectures = ['Qwen3ForCausalLM']
    if quant_config is not None:
        cfg.quant_config = quant_config

    class FakeModel:

        def __init__(self, config):
            self.config = config
            self.model = FakeConfig()
            self.model.config = config

        def to(self, *a, **k):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter(())

        def named_modules(self):
            return iter(())
    return FakeModel(cfg)
_Q3_14B_PROFILE = dict(hidden_size=5120, intermediate_size=17408, num_hidden_layers=40, num_key_value_heads=8, num_attention_heads=40)
_Q3_32B_PROFILE = dict(hidden_size=5120, intermediate_size=27648, num_hidden_layers=64, num_key_value_heads=8, num_attention_heads=64)
_Q3_8B_PROFILE = dict(hidden_size=4096, intermediate_size=12288, num_hidden_layers=36, num_key_value_heads=8, num_attention_heads=32)

def test_qwen3_14b_fp16_has_no_certified_profile():
    model = _make_fake_qwen3_model(**_Q3_14B_PROFILE)
    try:
        Qwen3Adapter(model)
    except UnsupportedModelError as e:
        msg = str(e)
        assert 'hidden_size=5120' in msg, f'Message must name the unsupported size: {msg}'
        assert 'no certified device execution profile' in msg, msg
        assert 'not a claim' in msg, msg
        assert 'docs/api_reference.md' in msg, f'Message must reference docs/api_reference.md: {msg}'
    else:
        raise AssertionError('D-12 envelope guard did NOT fire on Qwen3-14B profile')

def test_qwen3_14b_fp16_via_from_pretrained_raises():
    from unittest.mock import patch, MagicMock
    fake_14b = _make_fake_qwen3_model(**_Q3_14B_PROFILE)
    with patch('rpu_backend.api.causal_lm.AutoConfig') as MockAutoConfig, patch('rpu_backend.api.causal_lm.AutoModelForCausalLM') as MockAutoModel:
        MockAutoConfig.from_pretrained.return_value = fake_14b.config
        MockAutoModel.from_pretrained.return_value = fake_14b
        from rpu_backend import RPUModelForCausalLM
        import torch
        try:
            RPUModelForCausalLM.from_pretrained('Qwen/Qwen3-14B', dtype=torch.float16)
        except UnsupportedModelError as e:
            msg = str(e)
            assert 'hidden_size=5120' in msg, f'E2E from_pretrained must surface adapter message: {msg}'
            assert 'docs/api_reference.md' in msg, f'E2E msg must reference docs: {msg}'
            MockAutoConfig.from_pretrained.assert_called_once()
            MockAutoModel.from_pretrained.assert_not_called()
        else:
            raise AssertionError('RPUModelForCausalLM.from_pretrained did NOT raise UnsupportedModelError on 14B fake config')

def test_qwen3_14b_w8a16_preflight_is_allowed():
    model = _make_fake_qwen3_model(**_Q3_14B_PROFILE, quant_config={'method': 'w8a16'})
    Qwen3Adapter.preflight(model.config)

def test_qwen3_14b_w8a16_via_from_pretrained_skips_fp16_load():
    from unittest.mock import patch
    fake_14b = _make_fake_qwen3_model(**_Q3_14B_PROFILE, quant_config={'method': 'w8a16'})
    with patch('rpu_backend.api.causal_lm.AutoConfig') as MockAutoConfig, patch('rpu_backend.api.causal_lm.AutoModelForCausalLM') as MockAutoModel, patch('rpu_backend.quant.load.load_w8a16_model') as MockW8A16Loader:
        MockAutoConfig.from_pretrained.return_value = fake_14b.config
        MockW8A16Loader.return_value = fake_14b
        from rpu_backend import RPUModelForCausalLM
        import torch
        model = RPUModelForCausalLM.from_pretrained('local/qwen3-14b-w8a16-lmhead-int8', dtype=torch.float16)
        assert model is fake_14b
        MockAutoConfig.from_pretrained.assert_called_once()
        MockAutoModel.from_pretrained.assert_not_called()
        MockW8A16Loader.assert_called_once()

def test_qwen3_8b_invalid_exact_chunk_fails_before_weight_load():
    from unittest.mock import patch
    fake_8b = _make_fake_qwen3_model(**_Q3_8B_PROFILE)
    with patch('rpu_backend.api.causal_lm.AutoConfig') as MockAutoConfig, patch('rpu_backend.api.causal_lm.AutoModelForCausalLM') as MockAutoModel:
        MockAutoConfig.from_pretrained.return_value = fake_8b.config
        from rpu_backend import RPUModelForCausalLM
        import torch
        with pytest.raises(UnsupportedModelError, match='chunk ceiling 128'):
            RPUModelForCausalLM.from_pretrained('local/qwen3-8b', dtype=torch.float16, rpu_execution={'prefill': {'chunk_size': 224}})
        MockAutoConfig.from_pretrained.assert_called_once()
        MockAutoModel.from_pretrained.assert_not_called()

def test_qwen3_32b_has_no_certified_profile():
    model = _make_fake_qwen3_model(**_Q3_32B_PROFILE)
    try:
        Qwen3Adapter(model)
    except UnsupportedModelError as e:
        msg = str(e)
        assert 'intermediate_size=27648' in msg or 'hidden_size=5120' in msg, f'Q32B profile must appear in error: {msg}'
    else:
        raise AssertionError('D-12 + MEDIUM-10 envelope guard did NOT fire on Qwen3-32B profile')

def test_supported_envelope_unchanged():
    assert (1024, 3072, 28, 8) in _SUPPORTED_PROFILES, '0.6B profile must be supported'
    assert (2048, 6144, 28, 8) in _SUPPORTED_PROFILES, '1.7B profile must be supported'
    assert (2560, 9728, 36, 8) in _SUPPORTED_PROFILES, '4B profile must be supported'
    assert (5120, 17408, 40, 8) not in _SUPPORTED_PROFILES, '14B must not be fp16-supported'
    assert (5120, 17408, 40, 8) in _W8A16_ONLY_PROFILES, '14B is W8A16-only'
    q3_8b_profile = (4096, 12288, 36, 8)
    assert q3_8b_profile in _SUPPORTED_PROFILES, '8B FP16 must remain certified'

def main() -> int:
    test_qwen3_14b_fp16_has_no_certified_profile()
    print('[PASS] Qwen3-14B FP16 has no certified profile')
    test_qwen3_32b_has_no_certified_profile()
    print('[PASS] Qwen3-32B has no certified profile')
    test_qwen3_14b_w8a16_preflight_is_allowed()
    print(f'[PASS] Qwen3-14B W8A16 preflight is allowed')
    test_qwen3_14b_w8a16_via_from_pretrained_skips_fp16_load()
    print(f'[PASS] Qwen3-14B W8A16 from_pretrained uses W8A16 loader path')
    test_supported_envelope_unchanged()
    print('[PASS] Certified FP16/W8A16 profile sets are intact')
    test_qwen3_14b_fp16_via_from_pretrained_raises()
    print(f'[PASS] MEDIUM-9 — RPUModelForCausalLM.from_pretrained dispatch raises')
    print(f'[PASS] D-12 14B/32B runtime guard acceptance — Plan 02-06 Task 3')
    return 0
if __name__ == '__main__':
    sys.exit(main())
