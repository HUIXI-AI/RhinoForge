"""Exercise the real Expert binder without creating a model or an device owner."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.hy_vla.nvfp4 import HyVlaExpertNvfp4Metadata
from rpu_backend.adapters.hy_vla.runtime import HyVlaRunner, _HyVlaNativeColdSnapshot
from rpu_backend.adapters.hy_vla.weights import HyVlaConfig


@pytest.mark.parametrize("precision", ["nvfp4", "w8", "fp16"])
def test_expert_binder_selects_native_setter_and_preserves_scale_order(monkeypatch, precision):
    projections = ("q", "k", "v", "o", "gate", "up", "down")
    events = []

    def spy(name):
        def call(*args):
            events.append((name, args))
        return call

    for suffix in ("create", "destroy", "set_runtime_config", "set_weights",
                   "set_weights_w8a16", "set_weights_nvfp4", "set_action_weights",
                   "set_gateup_resident_weights"):
        name = "hyvla_expert_" + suffix
        monkeypatch.setattr(torch.ops.rpu, name, spy(name), raising=False)

    expert = {name: [torch.tensor([index], dtype=torch.float16)]
              for index, name in enumerate(("q_w", "k_w", "v_w", "o_w", "q_norm",
                  "k_norm", "input_norm", "post_norm", "gate_w", "up_w", "down_w"))}
    for index, projection in enumerate(projections):
        expert[projection + "_ws"] = ([] if precision == "fp16" else
            [torch.tensor([index], dtype=torch.uint8 if precision == "nvfp4" else torch.float16)])
        if precision == "nvfp4":
            expert[projection + "_ts"] = [torch.full((32,), index + 0.5, dtype=torch.float32)]
    if precision == "nvfp4":
        expert["gateup_resident"] = [torch.empty(1, dtype=torch.float16)]

    constants = tuple(torch.tensor([i], dtype=torch.float16) for i in range(3))
    unroll_weights = tuple(torch.tensor([i], dtype=torch.float16) for i in range(6))
    bound = object()

    def bind_sub(name, create, destroy, binder, signature, *, variant):
        assert name == "expert" and variant is True
        binder(37)
        return bound

    owner = SimpleNamespace(
        w=SimpleNamespace(cfg=HyVlaConfig(), expert=expert, expert_final_norm=torch.ones(1),
            expert_nvfp4=HyVlaExpertNvfp4Metadata(resident_layers=12) if precision == "nvfp4" else None,
            kv_heads_rpu=8, action_mlp_cores=8, attn_tp=8, expert_o_resident_layers=0),
        prefix_len=240, _native_cold=_HyVlaNativeColdSnapshot.from_environment({}),
        _unroll=True, _uw=unroll_weights, _sub=bind_sub, _component_cores=lambda component: 8,
        _denoise_constants=lambda *args: constants)

    result, returned_constants = HyVlaRunner._prepare_expert(owner, 291, None, None, None)
    assert result is bound and all(a is b for a, b in zip(returned_constants, constants))
    setter = "hyvla_expert_set_weights" + {"nvfp4": "_nvfp4", "w8": "_w8a16", "fp16": ""}[precision]
    expected_order = ["hyvla_expert_set_runtime_config", setter, "hyvla_expert_set_action_weights"]
    if precision == "nvfp4":
        expected_order.append("hyvla_expert_set_gateup_resident_weights")
    assert [name for name, _ in events] == expected_order
    args = events[1][1]
    assert args[0] == 37 and args[1] is expert["q_w"] and args[11] is expert["down_w"]
    assert args[12] is constants[0] and args[13] is constants[1]
    assert args[14] is owner.w.expert_final_norm
    assert args[15:23] == (16, 8, 128, 1024, 2048, 1e-5, 51, 8)
    if precision == "fp16":
        assert len(args) == 23
    else:
        assert all(actual is expert[p + "_ws"] for actual, p in zip(args[23:30], projections))
        assert len(args) == (37 if precision == "nvfp4" else 30)
        if precision == "nvfp4":
            assert all(actual is expert[p + "_ts"][0] for actual, p in zip(args[30:], projections))
            assert events[-1][1][1] is expert["gateup_resident"][0]
    assert all(actual is expected for actual, expected in zip(events[2][1][1:7], unroll_weights))
