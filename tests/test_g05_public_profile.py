from __future__ import annotations

from types import SimpleNamespace

import pytest

from rpu_backend.adapters.g05 import runtime as g05


class _Policy:
    training = False
    continuous_action = True
    discrete_action = True
    predict_cot = True
    return_continuous_action = True

    def __init__(self) -> None:
        self.processor = SimpleNamespace(
            batchify_action=False,
            decode_text=lambda *_args, **_kwargs: ["cot"],
            decode_ar=lambda *_args, **_kwargs: (["ar"], [], None, []),
        )
        self.model = SimpleNamespace(
            training=False,
            use_training_rtc=False,
            action_expert=SimpleNamespace(training=False, config=object()),
            ar_helper=SimpleNamespace(
                block_wise_autoregressive=False,
                do_sample=False,
                max_new_tokens=300,
                eov_token_id=7,
                _token_index_ranges=[{"lower": 10, "upper": 20, "pos_id": 3}],
            ),
            fm_helper=SimpleNamespace(
                horizon_steps=32,
                action_dim=27,
                num_inference_steps=10,
                time_convention="pi_convention",
                use_correlated_noise=False,
                action_causal=False,
                _sample_noise=lambda: None,
            ),
            cfg=SimpleNamespace(ae_vlm_condition_mode="cross_attn_only"),
            vlm=SimpleNamespace(),
            inference_fm=lambda *_args, **_kwargs: None,
            inference_ar=lambda *_args, **_kwargs: None,
        )
        self.forward_calls = []

    def parameters(self):
        return ()

    def predict_action(self, batch):
        raise AssertionError("the RPU wrapper was not installed")

    def generate_text(self, *_args, **_kwargs):
        return None

    def generate_action(self, *_args, **_kwargs):
        return None

    def forward_inference(self, **kwargs):
        self.forward_calls.append(kwargs)
        return {
            "action": "continuous",
            "ar_action": "discrete",
            "cot_text": ["cot"],
        }


def _isolate_install(monkeypatch, claims):
    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    monkeypatch.setattr(g05, "_claim_live_instance", claims.append)
    monkeypatch.setattr(g05, "_preflight_g05_vlm", lambda model: model.vlm)
    monkeypatch.setattr(g05, "_check_g05_action_profile", lambda _config: None)
    monkeypatch.setattr(
        g05,
        "patch_g05_vlm_for_rpu",
        lambda model, *, max_seq_len: model,
    )
    monkeypatch.setattr(
        g05,
        "patch_g05_action_expert_for_rpu",
        lambda expert, *, max_seq_len: expert,
    )


def test_g05_exact_public_modes_delegate_to_upstream_forward(monkeypatch) -> None:
    claims = []
    _isolate_install(monkeypatch, claims)
    policy = _Policy()

    assert g05.patch_g05_policy_for_rpu(policy) is policy
    batch = {
        "samples": ["sample"],
        "pixel_values": "pixels",
        "action": "golden",
        "action_dim_is_pad": "pad-mask",
    }
    assert policy.predict_action(batch) == {
        **batch,
        "action": "continuous",
        "ar_action": "discrete",
        "cot_text": ["cot"],
    }
    assert policy.forward_calls == [
        {
            "samples": ["sample"],
            "pixel_values": "pixels",
            "actions": "golden",
            "action_dim_is_pad": "pad-mask",
        }
    ]
    assert claims == [policy.model]


def test_g05_non_exact_modes_fail_before_rpu_claim(monkeypatch) -> None:
    invalid = (
        ("continuous_action", False),
        ("discrete_action", False),
        ("predict_cot", False),
        ("return_continuous_action", False),
    )
    for name, value in invalid:
        claims = []
        _isolate_install(monkeypatch, claims)
        policy = _Policy()
        setattr(policy, name, value)
        with pytest.raises(NotImplementedError, match="pinned public base profile"):
            g05.patch_g05_policy_for_rpu(policy)
        assert claims == []
        assert not hasattr(policy, "_rpu_g05_policy_install_started")

    claims = []
    _isolate_install(monkeypatch, claims)
    policy = _Policy()
    policy.model.ar_helper.block_wise_autoregressive = True
    with pytest.raises(NotImplementedError, match="greedy non-BAR"):
        g05.patch_g05_policy_for_rpu(policy)
    assert claims == []

    with pytest.raises(ValueError, match="max_seq_len=2048"):
        g05.patch_g05_policy_for_rpu(_Policy(), max_seq_len=1024)
    assert claims == []
