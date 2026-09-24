"""Board-free contracts for HyVLA's bounded checkpoint materialization."""

from __future__ import annotations

from collections.abc import Iterable
import math

import pytest
import torch

from rpu_backend.adapters.hy_vla import runtime as hyvla_runtime
from rpu_backend.adapters.hy_vla import weights as hyvla_weights
from rpu_backend.api.hy_embodied import HyEmbodiedPolicy


class _FakeSafeTensorHandle:
    def __init__(self, keys: Iterable[str], values=None):
        self._keys = tuple(keys)
        self._values = values or {}
        self.calls = []
        self.slice_calls = []
        self.events = []
        self.active = False
        self.enter_calls = 0

    def __enter__(self):
        assert not self.active
        self.enter_calls += 1
        self.active = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.active = False

    def keys(self):
        return self._keys

    def get_tensor(self, name):
        assert self.active, "safetensors handle closed before lazy materialization"
        self.calls.append(name)
        self.events.append(("get", name))
        if name not in self._values:
            raise AssertionError(f"unexpected materialization of {name}")
        return self._values[name].clone()

    def get_slice(self, name):
        self.slice_calls.append(name)
        if name not in self._values:
            raise AssertionError(f"unexpected metadata access for {name}")
        return _FakeTensorSlice(self._values[name])


class _FakeTensorSlice:
    def __init__(self, tensor):
        self._tensor = tensor

    def get_dtype(self):
        return {
            torch.bfloat16: "BF16",
            torch.float16: "F16",
            torch.float32: "F32",
        }[self._tensor.dtype]

    def get_shape(self):
        return list(self._tensor.shape)


def _tiny_cfg():
    return hyvla_weights.HyVlaConfig(
        vit_layers=1,
        vit_heads=1,
        vit_hidden=2,
        vit_head_dim=2,
        vit_inter=2,
        vit_seq=1,
        layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=1,
        vlm_hidden=2,
        vlm_inter=2,
        expert_hidden=2,
        expert_inter=2,
        proj_dim=2,
        action_dim=2,
    )


def _bf16_values(keys, cfg):
    shapes = hyvla_weights._expected_checkpoint_shapes(cfg)
    return {
        key: torch.arange(math.prod(shapes[key])).reshape(shapes[key]).to(
            torch.bfloat16)
        for key in keys
    }


def _clear_hyvla_cold_environment(monkeypatch):
    keys = set(hyvla_runtime._HY_VLA_ENV_DEFAULTS)
    keys.update({
        "RPU_HY_VLA_ACTION_MLP_MC",
        "RPU_HY_VLA_CACHING_ALLOC",
        "RPU_HY_VLA_MOT_NORM_NOMERGE",
        "RPU_HY_VLA_PREFIX_TEMPLATE",
        "RPU_HY_VLA_PROJ1_IN_MERGER",
        "RPU_HY_VLA_RMSNORM_PAD16",
        "RPU_HY_VLA_W4A16",
        "RPU_HY_VLA_W8A16",
    })
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_supported_checkpoint_inventory_accounts_for_all_tensors():
    cfg = hyvla_weights.HyVlaConfig()
    expected = hyvla_weights._expected_checkpoint_keys(cfg)
    unused = hyvla_weights._unused_expert_checkpoint_keys(cfg)

    assert len(expected) == 1628
    assert len(unused) == 32 * 11 == 352
    assert unused < expected
    assert len(hyvla_weights._expected_materialized_keys(
        cfg, load_pos_embedding=True)) == 1276
    assert len(hyvla_weights._expected_materialized_keys(
        cfg, load_pos_embedding=False)) == 1275


def test_lazy_view_converts_each_access_to_float32_without_caching():
    full_name = "prefix.tensor"
    source = torch.tensor([1.25, -2.5], dtype=torch.bfloat16)
    handle = _FakeSafeTensorHandle((full_name,), {full_name: source})
    seen = set()

    with handle:
        view = hyvla_weights._LazyFloat32TensorView(
            handle, "prefix.", (full_name,), seen)
        assert handle.calls == []
        first = view["tensor"]
        second = view["tensor"]

    assert first.dtype == second.dtype == torch.float32
    assert torch.equal(first, source.float())
    assert torch.equal(second, source.float())
    assert first.data_ptr() != second.data_ptr()
    assert handle.calls == [full_name, full_name]
    assert seen == {full_name}
    with pytest.raises(KeyError, match="checkpoint has no tensor"):
        view["missing"]


@pytest.mark.parametrize(
    "inventory_error", ["missing required", "missing unused", "unexpected"])
def test_inventory_mismatch_rejects_before_any_tensor_materialization(
        monkeypatch, inventory_error):
    cfg = _tiny_cfg()
    keys = hyvla_weights._expected_checkpoint_keys(cfg)
    if inventory_error == "missing required":
        keys.remove(hyvla_weights._LM_HEAD)
    elif inventory_error == "missing unused":
        keys.remove(next(iter(hyvla_weights._unused_expert_checkpoint_keys(cfg))))
    else:
        keys.add("model.unsupported.weight")
    handle = _FakeSafeTensorHandle(sorted(keys))
    monkeypatch.setattr(hyvla_weights, "safe_open", lambda *args, **kwargs: handle)

    with pytest.raises(ValueError, match=inventory_error.split()[0]):
        hyvla_weights.load_hy_vla_weights(
            "unused.safetensors", pos_embedding=torch.zeros(1, 1, 2), cfg=cfg)

    assert handle.calls == []
    assert handle.slice_calls == []
    assert not handle.active


def test_preflight_validates_every_metadata_entry_without_materializing(
        monkeypatch):
    cfg = _tiny_cfg()
    keys = hyvla_weights._expected_checkpoint_keys(cfg)
    handle = _FakeSafeTensorHandle(sorted(keys), _bf16_values(keys, cfg))
    monkeypatch.setattr(hyvla_weights, "safe_open", lambda *args, **kwargs: handle)

    assert hyvla_weights.preflight_hy_vla_checkpoint(
        "unused.safetensors", cfg) is None

    assert handle.calls == []
    assert set(handle.slice_calls) == keys
    assert len(handle.slice_calls) == len(keys)
    assert not handle.active


@pytest.mark.parametrize("metadata_error", ["dtype", "shape"])
def test_preflight_metadata_mismatch_rejects_before_materialization(
        monkeypatch, metadata_error):
    cfg = _tiny_cfg()
    keys = hyvla_weights._expected_checkpoint_keys(cfg)
    values = _bf16_values(keys, cfg)
    target = "model.action_in_proj.bias"
    if metadata_error == "dtype":
        values[target] = values[target].float()
    else:
        values[target] = torch.zeros(3, dtype=torch.bfloat16)
    handle = _FakeSafeTensorHandle(sorted(keys), values)
    monkeypatch.setattr(hyvla_weights, "safe_open", lambda *args, **kwargs: handle)

    with pytest.raises(ValueError, match="expected BF16"):
        hyvla_weights.preflight_hy_vla_checkpoint("unused.safetensors", cfg)

    assert handle.calls == []
    assert set(handle.slice_calls) == keys
    assert not handle.active


@pytest.mark.parametrize("override_pos_embedding", [False, True])
def test_loader_streams_fp32_sources_and_never_reads_dead_expert_branch(
        monkeypatch, override_pos_embedding):
    cfg = _tiny_cfg()
    keys = hyvla_weights._expected_checkpoint_keys(cfg)
    handle = _FakeSafeTensorHandle(sorted(keys), _bf16_values(keys, cfg))
    monkeypatch.setattr(hyvla_weights, "safe_open", lambda *args, **kwargs: handle)
    monkeypatch.delenv("RPU_HY_VLA_W4A16", raising=False)
    monkeypatch.delenv("RPU_HY_VLA_W8A16", raising=False)
    monkeypatch.delenv("RPU_HY_VLA_ATTN_TP8", raising=False)

    uploads = []

    def fake_rpu16(tensor):
        assert tensor.device.type == "cpu"
        uploads.append(("plain", tensor.dtype, tuple(tensor.shape)))
        handle.events.append(("transform", "plain"))
        return tensor.half().clone()

    def fake_quant(tensor, partition, cores, bits):
        assert tensor.dtype == torch.float32
        assert bits == 16
        uploads.append(("linear", partition, cores, tuple(tensor.shape)))
        handle.events.append(("transform", "linear"))
        return tensor.half().clone(), None

    def fake_col(tensor, cores):
        assert tensor.dtype == torch.float32
        uploads.append(("col", cores, tuple(tensor.shape)))
        handle.events.append(("transform", "col"))
        return tensor.half().clone()

    monkeypatch.setattr(hyvla_weights, "_rpu16", fake_rpu16)
    monkeypatch.setattr(hyvla_weights, "_quant", fake_quant)
    monkeypatch.setattr(hyvla_weights, "_col", fake_col)
    monkeypatch.setattr(
        hyvla_weights, "transform_linear_weight",
        lambda tensor, partition, num_cores: tensor.clone())

    pos_embedding = torch.zeros(1, 1, 2) if override_pos_embedding else None
    result = hyvla_weights.load_hy_vla_weights(
        "unused.safetensors", pos_embedding=pos_embedding, cfg=cfg)

    assert uploads
    assert set(handle.slice_calls) == keys
    assert result.tok_emb.dtype == torch.float32
    assert set(result.host) == set(hyvla_weights._HOST_TENSORS)
    assert all(tensor.dtype == torch.float32 for tensor in result.host.values())
    pos_key = hyvla_weights._VIT + "pos_embed"
    assert (pos_key in handle.calls) is not override_pos_embedding
    dead_expert = hyvla_weights._unused_expert_checkpoint_keys(cfg)
    assert dead_expert.isdisjoint(handle.calls)
    assert set(handle.calls) == hyvla_weights._expected_materialized_keys(
        cfg, load_pos_embedding=not override_pos_embedding)
    assert len(set(handle.calls)) < len(keys)
    first_transform = next(
        index for index, event in enumerate(handle.events)
        if event[0] == "transform")
    last_get = max(
        index for index, event in enumerate(handle.events) if event[0] == "get")
    assert first_transform < last_get
    assert not handle.active


def test_loader_rejects_malformed_quant_plan_before_opening_checkpoint(
        monkeypatch):
    _clear_hyvla_cold_environment(monkeypatch)
    monkeypatch.setenv("RPU_HY_VLA_W8A16", "expert,typo")
    handle = _FakeSafeTensorHandle(())
    monkeypatch.setattr(hyvla_weights, "safe_open", lambda *args, **kwargs: handle)

    with pytest.raises(ValueError, match="RPU_HY_VLA_W8A16.*typo"):
        hyvla_weights.load_hy_vla_weights("unused.safetensors")

    assert handle.enter_calls == 0
    assert handle.calls == []
    assert handle.slice_calls == []


def test_weight_cold_plan_freezes_mixed_quantization_and_layout():
    plan = hyvla_weights._HyVlaWeightColdPlan.from_environment({
        "RPU_HY_VLA_W4A16": "vlm_text",
        "RPU_HY_VLA_W8A16": "all",
        "RPU_HY_VLA_ATTN_TP8": "1",
        "RPU_HY_VLA_PATCH_EMBED_MC": "0",
        "RPU_HY_VLA_ACTION_MLP_MC": "off",
    })

    assert plan.vit_bits == 8
    assert plan.vlm_text_bits == 4
    assert plan.vlm_vision_bits == 8
    assert plan.expert_bits == 8
    assert plan.attn_tp8 is True
    assert plan.patch_embed_cores == 1
    assert plan.action_mlp_cores == 1


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("RPU_RMSNORM_VWARP", "wide", "RPU_RMSNORM_VWARP"),
        ("RPU_HY_VLA_FAST_REPLAY", "vit,typo", "unknown components"),
        ("RPU_HY_VLA_MOT_NORM_NOMERGE", "qkv,typo", "unknown components"),
        ("RPU_HY_VLA_W8A16", "expert,typo", "不认识的名字"),
    ],
)
def test_direct_build_rejects_malformed_cold_plan_before_lifecycle_or_checkpoint(
        monkeypatch, tmp_path, key, value, match):
    _clear_hyvla_cold_environment(monkeypatch)
    monkeypatch.setenv(key, value)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    handle = _FakeSafeTensorHandle(())
    events = []

    monkeypatch.setattr(hyvla_weights, "safe_open", lambda *args, **kwargs: handle)
    monkeypatch.setattr(
        "rpu_backend.api.causal_lm._claim_live_instance",
        lambda _owner: events.append("claim"),
    )
    monkeypatch.setattr(
        "rpu_backend.api.causal_lm._poison_live_instance",
        lambda *_args: events.append("poison"),
    )
    monkeypatch.setattr(
        torch.rpu,
        "set_caching_allocator",
        lambda _enabled: events.append("allocator"),
    )

    with pytest.raises(ValueError, match=match):
        hyvla_runtime.build_hy_vla(checkpoint)

    assert events == []
    assert handle.enter_calls == 0
    assert handle.calls == []
    assert handle.slice_calls == []


def test_build_reuses_frozen_cold_plan_after_environment_changes(
        monkeypatch, tmp_path):
    from types import SimpleNamespace

    _clear_hyvla_cold_environment(monkeypatch)
    plan = hyvla_runtime.prepare_hy_vla_cold_plan({
        "RPU_HY_VLA_FAST_REPLAY": "vlm",
        "RPU_HY_VLA_W8A16": "expert",
        "RPU_RMSNORM_VWARP": "16",
    })
    # A plan is the sole authority after admission. These values would all fail
    # if build or loader parsed the process environment a second time.
    monkeypatch.setenv("RPU_HY_VLA_FAST_REPLAY", "bad-component")
    monkeypatch.setenv("RPU_HY_VLA_W8A16", "bad-quant-token")
    monkeypatch.setenv("RPU_RMSNORM_VWARP", "bad-rms-route")

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    events = []
    captured = {}

    monkeypatch.setattr(
        hyvla_runtime,
        "preflight_hy_vla_checkpoint",
        lambda path, cfg: events.append(("preflight", path, cfg)),
    )
    monkeypatch.setattr(
        "rpu_backend.api.causal_lm._claim_live_instance",
        lambda _owner: events.append("claim"),
    )
    monkeypatch.setattr(
        "rpu_backend.api.causal_lm._poison_live_instance",
        lambda *_args: events.append("poison"),
    )
    monkeypatch.setattr(
        torch.rpu,
        "set_caching_allocator",
        lambda enabled: events.append(("allocator", enabled)),
    )

    def fake_load(path, pos_embedding, cfg, *, _cold_plan):
        events.append("load")
        captured["weight_plan"] = _cold_plan
        return SimpleNamespace(cfg=cfg)

    def fake_runner(**kwargs):
        events.append("runner")
        captured["runner_kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(hyvla_runtime, "load_hy_vla_weights", fake_load)
    monkeypatch.setattr(hyvla_runtime, "HyVlaRunner", fake_runner)

    result = hyvla_runtime.build_hy_vla(checkpoint, _cold_plan=plan)

    assert isinstance(result, SimpleNamespace)
    assert [event if isinstance(event, str) else event[0] for event in events] == [
        "preflight", "claim", "allocator", "poison", "load", "runner"
    ]
    assert captured["weight_plan"] is plan.weights
    assert captured["runner_kwargs"]["_native_cold"] is plan.native
    assert captured["runner_kwargs"]["_unroll"] is plan.runner.unroll
    assert plan.weights.expert_bits == 8
    assert plan.native.vlm.fast_replay is True
    assert plan.native.vit.fast_replay is False
    assert plan.native.vlm.rmsnorm_capability == 3


def test_direct_build_preflights_before_process_claim(
        monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    events = []

    def reject_preflight(path, cfg):
        events.append(("preflight", path, cfg))
        raise ValueError("synthetic checkpoint schema rejection")

    def forbidden_claim(_owner):
        events.append(("claim",))
        raise AssertionError("claim ran before host preflight")

    monkeypatch.setattr(
        hyvla_runtime, "preflight_hy_vla_checkpoint", reject_preflight)
    monkeypatch.setattr(
        "rpu_backend.api.causal_lm._claim_live_instance", forbidden_claim)
    monkeypatch.setattr(
        "rpu_backend.api.causal_lm._poison_live_instance", forbidden_claim)
    monkeypatch.setattr(
        hyvla_runtime, "load_hy_vla_weights", forbidden_claim)

    with pytest.raises(ValueError, match="schema rejection"):
        hyvla_runtime.build_hy_vla(checkpoint)

    assert [event[0] for event in events] == ["preflight"]


def test_policy_preflights_before_claim_or_materialization_marker(
        monkeypatch, tmp_path):
    for name in (
            "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json"):
        (tmp_path / name).touch()
    policy = HyEmbodiedPolicy.from_checkpoint(tmp_path)
    events = []

    def reject_preflight(path, cfg=hyvla_weights.HyVlaConfig()):
        events.append(("preflight", path, cfg))
        raise ValueError("synthetic policy checkpoint schema rejection")

    def forbidden_claim(_owner):
        events.append(("claim",))
        raise AssertionError("claim ran before host preflight")

    monkeypatch.setattr(
        hyvla_weights, "preflight_hy_vla_checkpoint", reject_preflight)
    monkeypatch.setattr(
        "rpu_backend.api.causal_lm._claim_live_instance", forbidden_claim)
    monkeypatch.setattr(
        "rpu_backend.api.causal_lm._poison_live_instance", forbidden_claim)
    monkeypatch.setattr(hyvla_runtime, "build_hy_vla", forbidden_claim)

    with pytest.raises(ValueError, match="schema rejection"):
        policy.to("rpu")

    assert [event[0] for event in events] == ["preflight"]
    assert policy._rpu_build_started is False
    assert policy._rpu_ready is False
    assert not policy._rpu_build_lock.locked()


def test_policy_malformed_checkpoint_inventory_fails_before_lifecycle_or_tensor(
        monkeypatch, tmp_path):
    _clear_hyvla_cold_environment(monkeypatch)
    for name in (
            "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json"):
        (tmp_path / name).touch()
    policy = HyEmbodiedPolicy.from_checkpoint(tmp_path)
    handle = _FakeSafeTensorHandle(())
    events = []
    monkeypatch.setattr(hyvla_weights, "safe_open", lambda *args, **kwargs: handle)
    monkeypatch.setattr(
        "rpu_backend.api.causal_lm._claim_live_instance",
        lambda _owner: events.append("claim"),
    )
    monkeypatch.setattr(
        "rpu_backend.api.causal_lm._poison_live_instance",
        lambda *_args: events.append("poison"),
    )

    with pytest.raises(ValueError, match="missing"):
        policy.to("rpu")

    assert events == []
    assert handle.enter_calls == 1
    assert handle.calls == []
    assert handle.slice_calls == []
    assert policy._rpu_build_started is False
    assert policy._rpu_ready is False
    assert not policy._rpu_build_lock.locked()


def test_policy_malformed_runtime_env_fails_before_claim_poison_or_checkpoint(
        monkeypatch, tmp_path):
    _clear_hyvla_cold_environment(monkeypatch)
    for name in (
            "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json"):
        (tmp_path / name).touch()
    policy = HyEmbodiedPolicy.from_checkpoint(
        tmp_path,
        runtime_env={"RPU_HY_VLA_W8A16": "expert,typo"},
    )
    handle = _FakeSafeTensorHandle(())
    events = []
    monkeypatch.setattr(hyvla_weights, "safe_open", lambda *args, **kwargs: handle)
    monkeypatch.setattr(
        "rpu_backend.api.causal_lm._claim_live_instance",
        lambda _owner: events.append("claim"),
    )
    monkeypatch.setattr(
        "rpu_backend.api.causal_lm._poison_live_instance",
        lambda *_args: events.append("poison"),
    )

    with pytest.raises(ValueError, match="RPU_HY_VLA_W8A16.*typo"):
        policy.to("rpu")

    assert events == []
    assert handle.enter_calls == 0
    assert handle.calls == []
    assert handle.slice_calls == []
    assert policy._rpu_build_started is False
    assert policy._rpu_ready is False
    assert not policy._rpu_build_lock.locked()


def test_policy_failure_after_materialization_marker_remains_terminal(
        monkeypatch, tmp_path):
    _clear_hyvla_cold_environment(monkeypatch)
    for name in (
            "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json"):
        (tmp_path / name).touch()
    policy = HyEmbodiedPolicy.from_checkpoint(tmp_path)
    events = []

    monkeypatch.setattr(
        hyvla_weights, "preflight_hy_vla_checkpoint", lambda *_args: None)
    monkeypatch.setattr(
        "rpu_backend.api.causal_lm._claim_live_instance",
        lambda owner: events.append(("claim", owner)),
    )
    monkeypatch.setattr(
        "rpu_backend.api.causal_lm._release_live_instance",
        lambda owner: events.append(("release", owner)),
    )

    def fail_after_marker(*_args, **kwargs):
        assert policy._rpu_build_started is True
        assert isinstance(kwargs["_cold_plan"], hyvla_runtime._HyVlaColdPlan)
        raise RuntimeError("synthetic materialization failure")

    monkeypatch.setattr(hyvla_runtime, "build_hy_vla", fail_after_marker)

    with pytest.raises(RuntimeError, match="materialization failure"):
        policy.to("rpu")

    assert events == [("claim", policy)]
    assert policy._rpu_build_started is True
    assert policy._rpu_ready is False
    assert not policy._rpu_build_lock.locked()
    from rpu_backend.api.errors import RPUBackendError

    with pytest.raises(RPUBackendError, match="先前在权重/RPU 物化开始后失败"):
        policy.to("rpu")
