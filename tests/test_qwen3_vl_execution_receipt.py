"""Real VL padding/decoder/cache/receipt flow; only native execution is doubled."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters import qwen3_vl as qvl
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime import decoder
from rpu_backend.runtime.execution_planner import (
    GRAPH_RETAINED_CACHE, plan_fixed_component_execution,
)


def _plan(logical=314, execution=320, chunk=128, position=0):
    return plan_fixed_component_execution(
        logical, execution_len=execution, chunk_size=chunk,
        component_id="language_model", stage="prefill", position=position,
        graph_mode=GRAPH_RETAINED_CACHE, physical_descriptor=(101, 202, 303),
    )


class _GraphCache:
    def __init__(self):
        self.signatures = []

    @contextmanager
    def capture(self, signature):
        self.signatures.append(signature)
        yield


@pytest.fixture
def flow(monkeypatch):
    state = {"native": [], "planned": []}
    native_to = torch.Tensor.to

    def cpu_device(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            args = ("cpu", *args[1:])
        if kwargs.get("device") == "rpu":
            kwargs = dict(kwargs, device="cpu")
        return native_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_device)
    def linear(x, weight, bias, partition, cores, acc32, scale):
        assert (partition, cores, acc32, scale) == (1, 8, False, None)
        return torch.nn.functional.linear(x, weight, bias)
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", linear, raising=False)
    text = torch.nn.Module()
    text._rpu_decoder_handle = 17
    text._rpu_text_graph_cache = _GraphCache()
    text._rpu_text_num_layers = 1
    text._rpu_text_deepstack_hash = 0
    text._rpu_text_hidden_size = 4
    text._rpu_decode_stage_descriptor = (404, 505)
    model = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4, vocab_size=8),
                               vision_config=SimpleNamespace(spatial_merge_size=2)),
        model=SimpleNamespace(language_model=text, visual=None, rope_deltas=None),
        lm_head=torch.nn.Linear(4, 8, bias=False).half(),
    )
    cache = RPUCache(1, 1, 384, 1, 16, device="cpu")
    state.update(model=model, text=text, cache=cache, plan=_plan())

    def choose(logical, physical_limit, padding_budget, **kwargs):
        result = state["plan"]
        state["planned"].append((logical, kwargs["position"]))
        kwargs["plan_result_sink"](result)
        return result.selected.execution_len, result.selected.stage_tuple.compute_chunk

    def forward(handle, hidden, keys, values, mask, position, causal,
                positions, visual, *rest):
        state["native"].append(dict(
            hidden=hidden.clone(), mask=None if mask is None else mask.clone(),
            positions=None if positions is None else positions.clone(),
            position=position, visual_rows=[value.shape[0] for value in visual],
            descriptor=tuple(rest[-1]),
        ))
        return hidden.clone() + 1

    monkeypatch.setattr(qvl, "_plan_causal_prefill_execution", choose)
    monkeypatch.setattr(qvl, "chunk_policy_key", lambda handle: 0)
    monkeypatch.setattr(qvl, "make_zero_visual_embeds",
                        lambda owner, length: [torch.zeros(length, 4).half()] * 3)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", forward, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_get_resolved_chunk_size",
                        lambda handle: state["plan"].selected.stage_tuple.compute_chunk
                        if state["native"][-1]["hidden"].shape[1] > 1 else 16,
                        raising=False)
    return state


def _vl_forward(flow, length, *, value=1):
    positions = torch.arange(flow["cache"].position, flow["cache"].position + length).view(1, 1, length).expand(3, 1, length)
    return qvl._rpu_qwen3vl_forward(
        flow["model"], inputs_embeds=torch.full((1, length, 4), value).half(),
        attention_mask=torch.ones(1, length, dtype=torch.int64),
        position_ids=positions, past_key_values=flow["cache"], logits_to_keep=1,
    )


def test_vl_padded_prefill_retains_logical_receipt_then_decodes_from_real_end(flow):
    result = _vl_forward(flow, 314)
    dispatched = flow["native"][0]
    assert dispatched["hidden"].shape == (1, 320, 4)
    assert torch.count_nonzero(dispatched["hidden"][:, 314:]) == 0
    # The causal path discards a redundant all-ones logical mask; physical
    # tail padding cannot affect preceding logical rows.
    assert dispatched["mask"] is None
    assert dispatched["positions"].shape == (320, 3)
    assert dispatched["visual_rows"] == [320, 320, 320]
    receipt = flow["text"]._rpu_last_execution_plan
    assert (receipt["logical_len"], receipt["execution_len"], receipt["padding_rows"], receipt["position"]) == (314, 320, 6, 0)
    assert receipt["physical_descriptor"] == dispatched["descriptor"] == (101, 202, 303)
    assert flow["cache"].position == 314
    assert result.logits.shape == (1, 1, 8)
    _vl_forward(flow, 1, value=2)
    assert flow["native"][-1]["position"] == 314
    assert flow["native"][-1]["descriptor"] == (404, 505)
    assert flow["cache"].position == 315
    assert flow["text"]._rpu_last_execution_plan["logical_len"] == 1


@pytest.mark.parametrize("length,position", [(320, 0), (16, 64)])
def test_vl_unpadded_initial_or_continuation_keeps_its_real_geometry(flow, length, position):
    flow["cache"].reset_to_position(position)
    flow["plan"] = _plan(logical=length, execution=length, chunk=16, position=position)
    _vl_forward(flow, length)
    receipt = flow["text"]._rpu_last_execution_plan
    assert (receipt["logical_len"], receipt["execution_len"], receipt["padding_rows"], receipt["position"]) == (length, length, 0, position)
    assert flow["cache"].position == position + length


@pytest.mark.parametrize("padded_input", [False, True])
def test_shared_runner_uses_the_same_plan_for_raw_and_pre_padded_inputs(flow, padded_input):
    length = 320 if padded_input else 314
    raw, cache = decoder._run_causal_decoder_forward(
        flow["text"], 17, inputs_embeds=torch.ones(1, length, 4).half(),
        past_key_values=flow["cache"], prefill_plan=(320, 128, flow["plan"]),
    )
    assert raw.shape == (1, 314, 4)
    assert cache.position == 314
    receipt = flow["text"]._rpu_last_execution_plan
    assert (receipt["logical_len"], receipt["execution_len"], receipt["padding_rows"]) == (314, 320, 6)


@pytest.mark.parametrize("field,value", [("logical_len", 320), ("execution_len", 336),
                                         ("chunk_size", 64), ("position", 1),
                                         ("physical_descriptor", ())])
def test_vl_still_rejects_a_corrupted_execution_receipt(flow, monkeypatch, field, value):
    real = qvl._run_causal_decoder_forward

    def corrupt(owner, *args, **kwargs):
        result = real(owner, *args, **kwargs)
        owner._rpu_last_execution_plan[field] = value
        return result

    monkeypatch.setattr(qvl, "_run_causal_decoder_forward", corrupt)
    with pytest.raises(RuntimeError, match=("no matching dispatched chunk receipt" if field == "chunk_size" else "observed metadata")):
        _vl_forward(flow, 314)


@pytest.mark.parametrize("length,execution,chunk", [(315, 320, 128), (320, 336, 128), (320, 320, 64)])
def test_shared_runner_rejects_shape_or_plan_drift_before_native(flow, length, execution, chunk):
    with pytest.raises(RuntimeError, match="input/physical plan mismatch"):
        decoder._run_causal_decoder_forward(
            flow["text"], 17, inputs_embeds=torch.ones(1, length, 4).half(),
            past_key_values=flow["cache"], prefill_plan=(execution, chunk, flow["plan"]),
        )
    assert flow["native"] == [] and flow["cache"].position == 0


def test_vl_still_rejects_a_different_dispatched_descriptor(flow, monkeypatch):
    real = qvl._run_causal_decoder_forward

    def corrupt(owner, *args, **kwargs):
        result = real(owner, *args, **kwargs)
        owner._rpu_last_execution_plan["physical_descriptor"] = (999,)
        return result

    monkeypatch.setattr(qvl, "_run_causal_decoder_forward", corrupt)
    with pytest.raises(RuntimeError, match="descriptor disagrees with its A6 winner"):
        _vl_forward(flow, 314)
