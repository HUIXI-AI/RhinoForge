"""Automatic VL positions retain padded rows through the real shared runner."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters import qwen3_vl
from rpu_backend.runtime.execution_planner import GRAPH_RETAINED_CACHE, plan_fixed_component_execution


def _vl_forward_with_real_shared_receipt(monkeypatch, logical, execution, position=0):
    """Keep root padding, shared dispatch/receipt and cache restoration real."""
    from rpu_backend.api.cache import RPUCache
    from rpu_backend.runtime import decoder
    from test_execution_planner import _native_candidate_v2

    chunk = 16
    descriptor = tuple(_native_candidate_v2(execution, (chunk,) * 3,
        logical_len=logical, position=position, routes=((1, 101, 2, 0, (), 0),)))
    plan = plan_fixed_component_execution(logical, execution_len=execution,
        chunk_size=chunk, component_id="language_model", stage="prefill",
        position=position, graph_mode=GRAPH_RETAINED_CACHE, physical_descriptor=descriptor)
    text = SimpleNamespace(config=SimpleNamespace(hidden_size=16),
        _rpu_decoder_handle=17, _rpu_decode_stage_descriptor=descriptor,
        _rpu_text_num_layers=1, _rpu_text_hidden_size=16, _rpu_text_deepstack_hash=0,
        _rpu_text_graph_cache=SimpleNamespace(capture=lambda _sig: nullcontext()),
        _rpu_execution={"prefill": {"padding_rows": 0} if position else {}},
        _rpu_deepstack_zero_keepalive=torch.zeros(8192, 16, dtype=torch.float16))
    model = SimpleNamespace(model=SimpleNamespace(language_model=text, visual=SimpleNamespace(), rope_deltas=None),
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=16, vocab_size=32),
                               vision_config=SimpleNamespace(spatial_merge_size=2)),
        lm_head=torch.nn.Linear(16, 32, bias=False, dtype=torch.float16))
    cache = RPUCache(1, 1, 512, 1, 16, device="cpu")
    cache.reset_to_position(position)
    embeddings = torch.arange(logical * 16, dtype=torch.float16).reshape(1, logical, 16) / 1024
    native_calls, shared_receipts, advanced_positions = [], [], []

    original_to = torch.Tensor.to
    def cpu_transfer(value, *args, **kwargs):
        if args and str(args[0]) == "rpu":
            args = ("cpu", *args[1:])
        if str(kwargs.get("device")) == "rpu":
            kwargs["device"] = "cpu"
        return original_to(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_transfer)
    def linear(x, weight, bias, partition, cores, acc32, scale):
        assert (partition, cores, acc32, scale) == (1, 8, False, None)
        return torch.nn.functional.linear(x, weight, bias)
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", linear, raising=False)
    monkeypatch.setattr(qwen3_vl, "chunk_policy_key", lambda _handle: 0)
    def prepared_plan(length, _limit, _budget, **kwargs):
        assert length == logical
        kwargs["plan_result_sink"](plan)
        return execution, chunk
    monkeypatch.setattr(qwen3_vl, "_plan_causal_prefill_execution", prepared_plan)

    def native(handle, hidden, _keys, _values, mask, start, _causal, rope_positions, dense, *args):
        assert handle == 17 and start == position
        assert tuple(hidden.shape) == (1, execution, 16)
        # A redundant all-ones logical mask is canonicalized away.
        assert mask is None
        assert tuple(rope_positions.shape) == (execution, 3)
        expected_positions = torch.arange(position, position + logical, dtype=torch.int32)[:, None].expand(-1, 3)
        if execution > logical:
            expected_positions = torch.cat((expected_positions,
                expected_positions[-1:].expand(execution - logical, -1)))
        assert torch.equal(rope_positions, expected_positions)
        assert all(tuple(value.shape) == (execution, 16) for value in dense)
        # The native manifest retains original P even though the shared Python
        # input is already E. Changing this descriptor is not the fix.
        assert tuple(args[-1]) == descriptor
        native_calls.append((int(hidden.shape[1]), tuple(args[-1])))
        return hidden + 1
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_get_resolved_chunk_size", lambda _handle: chunk, raising=False)
    def observed_shared(*args, **kwargs):
        result = decoder._run_causal_decoder_forward(*args, **kwargs)
        receipt = text._rpu_last_execution_plan
        shared_receipts.append(dict(receipt))
        advanced_positions.append(cache.position)
        return result
    monkeypatch.setattr(qwen3_vl, "_run_causal_decoder_forward", observed_shared)
    def run():
        return qwen3_vl._rpu_qwen3vl_forward.__wrapped__(model,
            inputs_embeds=embeddings, attention_mask=torch.ones(1, logical, dtype=torch.int64),
            position_ids=None, past_key_values=cache)
    return run, model, cache, plan, embeddings, native_calls, shared_receipts, advanced_positions


@pytest.mark.parametrize("logical,execution,position", [(17, 32, 0), (16, 16, 0), (1, 1, 16)])
def test_vl_automatic_positions_preserve_execution_rows(monkeypatch, logical, execution, position):
    run, model, cache, plan, embeddings, native_calls, shared, advanced = (
        _vl_forward_with_real_shared_receipt(monkeypatch, logical, execution, position))
    output = run()
    cache.reset_to_position(position)
    repeated = run()
    assert torch.equal(repeated.logits, output.logits)
    assert native_calls == [(execution, plan.selected.stage_tuple.physical_descriptor)] * 2
    assert cache.position == position + logical
    assert all(row["logical_len"] == logical and row["execution_len"] == execution for row in shared)
    assert advanced == [position + logical] * 2
    assert torch.equal(output.logits, model.lm_head(embeddings + 1))
