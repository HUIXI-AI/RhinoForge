"""Small CPU checks for public Pi profile admission and the NVFP4 loader ABI."""
import os
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.pi05.optimized import (
    normalize_profile, execution_for_profile, profile_environment_scope,
    validate_checkpoint, validate_profile_batch,
)
from rpu_backend.adapters.pi05.loader import (
    _pop_pi05_w8a16_tensors, _install_pi05_w8a16_tensors,
)
from rpu_backend.adapters.pi05.w4pack import pack_int4_projections_inplace
from rpu_backend.quant.convert_pi05 import _convert_tensor


def test_profile_restores_environment_and_rejects_conflicting_workload(monkeypatch):
    profile = normalize_profile(dict(precision="w8_action_nvfp4", num_cameras=2))
    monkeypatch.delenv("RPU_PI05_VISION_OWNER_LN", raising=False)
    with pytest.raises(RuntimeError, match="load failed"):
        with profile_environment_scope(profile):
            assert os.environ["RPU_PI05_VISION_OWNER_LN"] == "1"
            assert os.environ["RPU_PI05_PREFILL_GATEUP_W8A8"] == "0"
            raise RuntimeError("load failed")
    assert "RPU_PI05_VISION_OWNER_LN" not in os.environ
    assert execution_for_profile(profile, None)["prefill"]["chunk_size"] == 272
    with pytest.raises(ValueError, match="chunk_size"):
        execution_for_profile(profile, {"prefill": {"chunk_size": 400}})
    with pytest.raises(ValueError, match="method=nvfp4a16"):
        validate_checkpoint({"method": "w4a16"}, "w4a16")


def test_profile_rejects_padded_language_before_execution():
    profile = normalize_profile(dict(precision="fp16", num_cameras=2))
    names = ["observation.images.base", "observation.images.wrist"]
    policy = SimpleNamespace(_optimized_profile=profile,
        _lerobot_policy=SimpleNamespace(config=SimpleNamespace(image_features=names)))
    batch = {name: torch.zeros(1, 3, 16, 16) for name in names}
    batch["observation.language.tokens"] = torch.zeros(1, 32, dtype=torch.int64)
    batch["observation.language.attention_mask"] = torch.ones(1, 32, dtype=torch.bool)
    validate_profile_batch(policy, batch, 10)
    batch["observation.language.attention_mask"][0, 1] = False
    with pytest.raises(ValueError, match="right padding"):
        validate_profile_batch(policy, batch, 10)


@pytest.mark.parametrize("chunk_size", ["auto", 400])
def test_profile_asset_check_accepts_frozen_policy_execution(chunk_size):
    from rpu_backend.api._execution import _freeze_rpu_execution
    from rpu_backend.adapters.pi05.optimized import required_kernel_names

    profile = normalize_profile(dict(precision="w8_prefill_a8_action_nvfp4", num_cameras=3))
    frozen = _freeze_rpu_execution({
        "model": {"num_cores": 8}, "prefill": {"chunk_size": chunk_size},
        "action": {"linear_acc32": False},
    })
    resolved = execution_for_profile(profile, frozen)
    assert resolved["prefill"]["chunk_size"] == 400
    assert frozen["prefill"]["chunk_size"] == chunk_size
    names = required_kernel_names(profile, frozen)
    assert "pi05_denoise_gate_up_geglu_nvfp4_acc16_m320n64k128" in names
    resolved["prefill"]["chunk_size"] = 432
    assert frozen["prefill"]["chunk_size"] == chunk_size
    with pytest.raises(TypeError):
        frozen["prefill"]["chunk_size"] = 432


@pytest.mark.parametrize("entry_point", ["predict_action_chunk", "select_action"])
def test_hot_profile_uses_current_input_without_cold_environment(monkeypatch, entry_point):
    from rpu_backend.api.policy import Pi05Policy
    from rpu_backend.adapters.pi05 import optimized

    names = ["observation.images.base", "observation.images.wrist"]
    batch = {name: torch.zeros(1, 3, 16, 16) for name in names}
    batch["observation.language.tokens"] = torch.zeros(1, 64, dtype=torch.int64)
    mask = batch["observation.language.attention_mask"] = torch.ones(1, 64, dtype=torch.bool)
    calls = []

    def infer(current, **kwargs):
        assert current is batch
        calls.append(kwargs)
        return current["observation.language.attention_mask"].sum().reshape(1).float()

    policy = object.__new__(Pi05Policy)
    policy._rpu_ready = True
    policy._optimized_profile = normalize_profile(
        dict(precision="w8_action_nvfp4", num_cameras=2, text_tokens=64)
    )
    policy._lerobot_policy = SimpleNamespace(
        config=SimpleNamespace(image_features=names),
        predict_action_chunk=infer, select_action=infer,
    )

    def forbid_cold_environment(*args, **kwargs):
        raise AssertionError("hot inference must not enter a cold environment scope or write env")

    call = getattr(policy, entry_point)
    kwargs = {"num_steps": 10} if entry_point == "predict_action_chunk" else {}
    with monkeypatch.context() as hot:
        hot.setattr(optimized, "profile_environment_scope", forbid_cold_environment)
        hot.setattr(os, "putenv", forbid_cold_environment)
        hot.setattr(os, "unsetenv", forbid_cold_environment)
        assert call(batch, **kwargs).item() == 64
        mask[0, -1] = False
        assert call(batch, **kwargs).item() == 63
        mask[0, 1] = False
        with pytest.raises(ValueError, match="right padding"):
            call(batch, **kwargs)
    assert len(calls) == 2


def test_action_nvfp4_keeps_tensor_scale_fp32_and_packs_once():
    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    policy.model.paligemma_with_expert = torch.nn.Module()
    expert = torch.nn.Module()
    policy.model.paligemma_with_expert.gemma_expert = torch.nn.Module()
    policy.model.paligemma_with_expert.gemma_expert.model = expert
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.q_proj = torch.nn.Linear(256, 128, bias=False)
    layer.self_attn.k_proj = torch.nn.Linear(256, 128, bias=False)
    expert.layers = torch.nn.ModuleList([layer])
    prefix = "model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn."
    quant = dict(method="nvfp4a16", int4_components=["expert"],
                 mixed_int8_suffixes=["k_proj.weight", "v_proj.weight"])
    state = {}
    for name in ("q_proj", "k_proj"):
        values, *_ = _convert_tensor(prefix + name + ".weight",
            torch.linspace(-1, 1, 128 * 256).reshape(128, 256), bits=4,
            int8_keep_suffixes=tuple(quant["mixed_int8_suffixes"]), w4_format="nvfp4")
        state.update(values)
    consumed = _pop_pi05_w8a16_tensors(policy, state, quant)
    assert not state
    _install_pi05_w8a16_tensors(policy, consumed)
    q, k = layer.self_attn.q_proj, layer.self_attn.k_proj
    assert q.weight.dtype == torch.uint8 and q.weight_scale.dtype == torch.uint8
    assert q.tensor_scale.dtype == torch.float32 and q.tensor_scale.item() > 0
    assert k.weight.dtype == torch.int8 and k.weight_scale.dtype == torch.float16
    assert pack_int4_projections_inplace(expert, {"q_proj"}, attn_num_cores=8) == 1
    original = q.weight.clone()
    assert q.weight.shape == (128, 128)
    assert pack_int4_projections_inplace(expert, {"q_proj"}, attn_num_cores=8) == 0
    assert torch.equal(original, q.weight)
