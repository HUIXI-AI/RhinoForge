"""Public loader capacity crosses the real hardware-attribute lifecycle."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.api import causal_lm, conditional_generation
from rpu_backend.api.errors import RPUConfigError
from rpu_backend.runtime import decoder, hw_attrs


@pytest.mark.parametrize("family", ["qwen3", "llama"])
@pytest.mark.parametrize("immediate", [False, True])
def test_real_causal_adapter_accepts_loader_capacity_before_claim(monkeypatch, family, immediate):
    from rpu_backend.adapters import qwen3, llama

    if family == "qwen3":
        from rpu_backend.tests.acceptance.test_qwen3_w4_public_install import _model
        model = _model()  # Exact 0.6B geometry, metadata-only FakeTensor weights.
        adapter_module = qwen3
    else:
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.config = SimpleNamespace(
            architectures=["LlamaForCausalLM"], model_type="llama",
            hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
            num_attention_heads=32, num_key_value_heads=8)
        adapter_module = llama
    monkeypatch.setattr(causal_lm.AutoConfig, "from_pretrained", lambda *a, **kw: model.config)
    monkeypatch.setattr(causal_lm.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: model)
    monkeypatch.setattr(causal_lm, "require_native_backend", lambda: None)
    monkeypatch.setattr(adapter_module, "require_native_backend", lambda: None)
    reached = []

    def claim(owner):
        assert owner is model
        # Both actual adapters have already executed validate_preinstall here.
        assert decoder._rotary_table_capacity(None, owner.model) == 16384
        reached.append(owner)
        raise RuntimeError("host device-claim boundary")

    monkeypatch.setattr(adapter_module, "_claim_live_instance", claim)
    options = dict(max_seq_len=16384, device="rpu" if immediate else None)
    if immediate:
        with pytest.raises(RuntimeError, match="host device-claim boundary"):
            causal_lm.RPUModelForCausalLM.from_pretrained("unused", **options)
    else:
        loaded = causal_lm.RPUModelForCausalLM.from_pretrained("unused", **options)
        assert loaded is model and reached == []
        with pytest.raises(RuntimeError, match="host device-claim boundary"):
            loaded.to("rpu")
    assert reached == [model]
    for owner in (model, model.model):
        assert owner._rpu_requested_max_seq_len == 16384
        assert not hasattr(owner, "_rpu_max_seq_len")
        assert not getattr(owner, "_rpu_swizzle_started", False)


@pytest.mark.parametrize("architecture", [
    "Qwen3VLForConditionalGeneration", "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
])
@pytest.mark.parametrize("immediate", [False, True])
def test_conditional_loader_stages_capacity_accepted_by_real_preinstall(monkeypatch, architecture, immediate):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    config = SimpleNamespace(architectures=[architecture])
    calls = []

    class Adapter:
        @classmethod
        def load_pretrained(cls, path, **kwargs):
            assert "max_seq_len" not in kwargs
            return model

        def __init__(self, owner):
            assert owner is model
            model.to = lambda device: self.to_rpu()

        def to_rpu(self):
            hw_attrs.validate_preinstall(model)
            assert model._rpu_requested_max_seq_len == 32768
            assert not hasattr(model, "_rpu_max_seq_len")
            calls.append("preinstall")
            hw_attrs.validate_postinstall(model)
            hw_attrs.install_hw_attr_validator(model)
            return model

    monkeypatch.setattr(conditional_generation.AutoConfig, "from_pretrained", lambda *a, **kw: config)
    monkeypatch.setattr(conditional_generation, "get_adapter", lambda arch: Adapter)
    loaded = conditional_generation.RPUModelForConditionalGeneration.from_pretrained(
        "unused", max_seq_len=32768, device="rpu" if immediate else None)
    assert loaded is model
    if not immediate:
        assert calls == []
        assert loaded.to("rpu") is model
    assert calls == ["preinstall"]
    with pytest.raises(RPUConfigError, match="cold capacity request"):
        model._rpu_requested_max_seq_len = 65536


@pytest.mark.parametrize("invalid", [None, False, 0, -1, 16384.0, "16384"])
def test_preinstall_rejects_invalid_pending_capacity_on_child(invalid):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    model[0]._rpu_requested_max_seq_len = invalid
    with pytest.raises(RPUConfigError, match="positive integer"):
        hw_attrs.validate_preinstall(model)


@pytest.mark.parametrize("attribute", ["_rpu_max_seq_len", "_rpu_swizzled", "_rpu_decoder_handle"])
def test_pending_capacity_does_not_admit_stale_installed_state(attribute):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    model._rpu_requested_max_seq_len = 16384
    setattr(model[0], attribute, 16384)
    with pytest.raises(RPUConfigError, match=attribute):
        hw_attrs.validate_preinstall(model)


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("hot", [False, True])
@pytest.mark.parametrize("operation", ["write", "delete"])
def test_requested_and_installed_capacity_remain_immutable(child, hot, operation):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    owner = model[0] if child else model
    owner._rpu_requested_max_seq_len = 16384
    hw_attrs.validate_preinstall(model)
    owner._rpu_max_seq_len = 16384
    hw_attrs.validate_postinstall(model)
    hw_attrs.install_hw_attr_validator(model)
    if hot:
        hw_attrs.mark_first_forward_done(model)
    for attribute in ("_rpu_requested_max_seq_len", "_rpu_max_seq_len"):
        with pytest.raises(RPUConfigError, match=attribute):
            if operation == "write":
                setattr(owner, attribute, 32768)
            else:
                delattr(owner, attribute)
        assert getattr(owner, attribute) == 16384


def test_pending_capacity_cannot_be_added_to_an_installed_default():
    model = torch.nn.Linear(1, 1)
    hw_attrs.install_hw_attr_validator(model)
    with pytest.raises(RPUConfigError, match="cold capacity request"):
        model._rpu_requested_max_seq_len = 16384
