"""Exact weight ABI and fail-closed delivery shard loading."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from rpu_backend.adapters.qwen3_vl.delivery_load import _checkpoint_inventory
from rpu_backend.adapters.qwen3_vl import _quantize_pack_lm_head_w4_groupwise as bounded
from rpu_backend.quant.int4_pgrp_pack import (
    quantize_int4_group_wise, swizzle_pack_int4_pgrp, swizzle_int4_pgrp_scale,
)


def make_checkpoint(root):
    tensors = {"a.weight": torch.arange(32).reshape(4, 8).half(),
               "b.weight": torch.zeros(2, 8, dtype=torch.bfloat16)}
    mapping = {}
    for i, (name, tensor) in enumerate(tensors.items()):
        filename = f"part-{i}.safetensors"
        save_file({name: tensor}, root / filename)
        mapping[name] = filename
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    return tensors, mapping


def test_two_shards(tmp_path):
    tensors, mapping = make_checkpoint(tmp_path)
    assert _checkpoint_inventory(tmp_path, tensors) == mapping


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "dtype", "traversal", "absolute", "index_disagrees"])
def test_invalid_checkpoint(tmp_path, change):
    tensors, mapping = make_checkpoint(tmp_path)
    if change == "missing":
        del mapping["a.weight"]
    elif change == "extra":
        mapping["extra.weight"] = "part-0.safetensors"
    elif change == "shape":
        tensors["a.weight"] = torch.empty(5, 8)
    elif change == "dtype":
        save_file({"a.weight": torch.zeros(4, 8, dtype=torch.int8)}, tmp_path / "part-0.safetensors")
    elif change == "traversal":
        mapping["a.weight"] = "../part-0.safetensors"
    elif change == "absolute":
        mapping["a.weight"] = str(tmp_path / "part-0.safetensors")
    else:
        mapping["a.weight"] = "part-1.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError):
        _checkpoint_inventory(tmp_path, tensors)

@pytest.mark.parametrize("rows,hidden", [(128, 64), (2176, 128), (4224, 256), (4096, 2048)])
def test_pack_exact(rows, hidden):
    torch.manual_seed(123)
    weight = torch.randn(rows, hidden).half()
    weight[0].zero_()
    values, logical, _ = quantize_int4_group_wise(weight, 32)
    expected_weight = swizzle_pack_int4_pgrp(values, 1, 8).reshape(rows, hidden // 2)
    expected_scale = swizzle_int4_pgrp_scale(logical, 32, 1, 8)
    actual_weight, actual_scale = bounded(weight)
    assert torch.equal(actual_weight, expected_weight)
    assert torch.equal(actual_scale, expected_scale)


@pytest.mark.parametrize("shape,group", [((127, 64), 32), ((128, 65), 32), ((128, 64), 64), ((128,), 32)])
def test_rejected_shapes(shape, group):
    with pytest.raises(ValueError):
        bounded(torch.zeros(shape, dtype=torch.float16), group)


def _delivery_config():
    from types import SimpleNamespace as NS
    return NS(model_type="qwen3_vl", architectures=["Qwen3VLForConditionalGeneration"],
              tie_word_embeddings=True, image_token_id=151655, video_token_id=151656,
              vision_start_token_id=151652, vision_end_token_id=151653,
              text_config=NS(model_type="qwen3_vl_text", hidden_size=2048,
                  intermediate_size=6144, num_hidden_layers=28, num_attention_heads=16,
                  num_key_value_heads=8, head_dim=128, vocab_size=151936,
                  hidden_act="silu", rms_norm_eps=1e-6, attention_bias=False,
                  use_cache=True, tie_word_embeddings=True,
                  rope_parameters={"rope_type": "default", "rope_theta": 5000000.0,
                                   "mrope_interleaved": True, "mrope_section": [24, 20, 20]}),
              vision_config=NS(model_type="qwen3_vl", hidden_size=1024,
                  intermediate_size=4096, depth=24, num_heads=16, patch_size=16,
                  temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=2048,
                  deepstack_visual_indexes=[5, 11, 17], hidden_act="gelu_pytorch_tanh",
                  in_channels=3, num_position_embeddings=2304))


def test_delivery_exact_opt_in_and_semantic_guard(monkeypatch):
    from rpu_backend.adapters import qwen3_vl as q
    config = _delivery_config()
    monkeypatch.delenv(q._QWEN3_VL_DELIVERY_BLOCKED_ENV, raising=False)
    with pytest.raises(q.UnsupportedModelError, match="numeric-blocked"):
        q._validate_delivery_config(config)
    monkeypatch.setenv(q._QWEN3_VL_DELIVERY_BLOCKED_ENV, "true")
    with pytest.raises(q.UnsupportedModelError, match="numeric-blocked"):
        q._validate_delivery_config(config)
    monkeypatch.setenv(q._QWEN3_VL_DELIVERY_BLOCKED_ENV, "1")
    q._validate_delivery_config(config)
    config.text_config.rope_parameters["mrope_section"] = [20, 20, 24]
    with pytest.raises(q.UnsupportedModelError, match="semantics"):
        q._validate_delivery_config(config)


@pytest.mark.parametrize("field,value", [("chunk_size", 128), ("padding_rows", 16), ("padding_budget", 64)])
def test_delivery_execution_is_checked_before_loading(field, value):
    from rpu_backend.adapters import qwen3_vl as q
    execution = {stage: dict(fields) for stage, fields in q._normalize_delivery_execution().items()}
    assert execution["vision"]["chunk_size"] == 576
    execution["prefill"][field] = value
    with pytest.raises((q.UnsupportedModelError, ValueError), match="C192|conflicts"):
        q._normalize_delivery_execution(execution)


@pytest.mark.parametrize("position,length", [(0, 575), (575, 1), (580, 1), (576, 2)])
def test_delivery_position_or_length_rejects_before_transport(position, length):
    from types import SimpleNamespace as NS
    from rpu_backend.adapters import qwen3_vl as q
    model = NS(_qwen3_vl_delivery_profile=q._QWEN3_VL_DELIVERY_PROFILE)
    with pytest.raises(q.UnsupportedModelError, match="P576"):
        q._rpu_qwen3vl_forward(model, input_ids=torch.zeros((1, length), dtype=torch.long),
            past_key_values=NS(position=position, max_seq_len=608), logits_to_keep=1)


@pytest.mark.parametrize("capacity", [608, 16384, 32768])
def test_delivery_accepts_larger_cache_before_transport(capacity):
    from types import SimpleNamespace as NS
    from rpu_backend.adapters import qwen3_vl as q
    model = NS(_qwen3_vl_delivery_profile=q._QWEN3_VL_DELIVERY_PROFILE)
    # The unsupported loss sentinel follows profile admission and precedes any
    # native transport, so it proves large owners retain the exact decode path.
    with pytest.raises(NotImplementedError, match="labels"):
        q._rpu_qwen3vl_forward(model, input_ids=torch.zeros((1, 1), dtype=torch.long),
            past_key_values=NS(position=576, max_seq_len=capacity),
            logits_to_keep=1, labels=torch.zeros(1))


def test_delivery_graph_key_owns_each_scalar_decode_position():
    from types import SimpleNamespace as NS
    from rpu_backend.adapters import qwen3_vl as q
    # The source helper asks native chunk_policy_key; only that endpoint is a
    # host boundary. Real graph-signature construction retains the position.
    from unittest.mock import patch
    model = NS(_rpu_text_num_layers=28, _rpu_text_deepstack_hash=31,
               _rpu_decoder_handle=7, _rpu_text_hidden_size=2048,
               _rpu_text_fused_decode_position=True)
    def signature(**fields):
        return tuple(fields["dyn_dims"])
    with patch.object(q.rpu_backend.graph, "GraphSignature", side_effect=signature), patch.object(q, "chunk_policy_key", return_value=0):
        signatures = [q._qwen3_vl_text_graph_signature(model, 1, 1, pos, 0)
                      for pos in range(576, 580)]
    assert len(set(signatures)) == 4
    model._rpu_text_fused_decode_position = False
    with patch.object(q.rpu_backend.graph, "GraphSignature", side_effect=signature), patch.object(q, "chunk_policy_key", return_value=0):
        assert q._qwen3_vl_text_graph_signature(model, 1, 1, 576, 0) == q._qwen3_vl_text_graph_signature(model, 1, 1, 577, 0)


@pytest.mark.parametrize("fault", [None, "wrong_position", "empty"])
def test_explicit_decode_descriptor_reaches_original_runner_receipt(monkeypatch, fault):
    from types import SimpleNamespace as NS
    from rpu_backend.runtime import decoder
    from rpu_backend.api.cache import RPUCache
    from rpu_backend.runtime.execution_planner import _decode_native_stage_domain, PlannerRejectError
    from test_execution_planner import _native_candidate_v2
    owner = NS(_rpu_decode_stage_descriptor=(999,))
    cache = object.__new__(RPUCache)
    cache.position, cache.max_seq_len, cache.batch_size = 576, 608, 1
    cache.k_caches, cache.v_caches = [torch.zeros(1)], [torch.zeros(1)]
    wire = tuple(_native_candidate_v2(1, (16, 16, 16), position=577 if fault == "wrong_position" else 576,
                                    descriptor_version=3, routes=((1, 101, 2, 0, (), 0),)))
    if fault == "empty": wire = ()
    def native(handle, hidden, keys, values, mask, position, *rest):
        descriptor = rest[-1]
        _decode_native_stage_domain((1, 1, len(descriptor), *descriptor), execution_len=1,
            logical_len=1, position=position, graph_mode="RETAINED_CACHE")
        return hidden.clone()
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_get_resolved_chunk_size", lambda _: 1, raising=False)
    def run():
        return decoder._run_causal_decoder_forward(owner, 7,
            inputs_embeds=torch.ones(1, 1, 4, dtype=torch.float16), past_key_values=cache,
            decode_descriptor=wire)
    if fault:
        with pytest.raises((RuntimeError, ValueError, PlannerRejectError)):
            run()
        assert cache.position == 576 and not hasattr(owner, "_rpu_last_execution_plan")
    else:
        result, actual_cache = run()
        assert actual_cache is cache and cache.position == 577
        assert owner._rpu_last_execution_plan["physical_descriptor"] == wire
        assert owner._rpu_last_execution_plan["position"] == 576
        assert torch.equal(result, torch.ones_like(result))
    assert owner._rpu_decode_stage_descriptor == (999,)


def test_retired_route_environment_rejects_even_a_typed_profile(monkeypatch):
    from types import SimpleNamespace as NS
    from rpu_backend.adapters import qwen3_vl as q
    monkeypatch.setenv("RPU_QWEN3VL_DECODE_LM_HEAD_W4A16", "1")
    with pytest.raises(q.UnsupportedModelError, match="retired"):
        q._lm_head_optimization_config(NS(_qwen3_vl_delivery_profile=q._QWEN3_VL_DELIVERY_PROFILE))


def test_cached_deepstack_prefix_observes_inplace_source_and_mask_changes(monkeypatch):
    from types import SimpleNamespace as NS
    from rpu_backend.adapters.qwen3_vl.text import scatter_visual_embeds_to_dense
    calls = []
    def scatter(destination, indices, parts):
        calls.append(indices.tolist())
        destination.index_copy_(0, indices, torch.cat(parts))
        return destination
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_scatter_row_parts_unflushed_", scatter, raising=False)
    owner = NS()
    prefix, suffix = torch.ones(1, 2).half(), torch.full((1, 2), 2.).half()
    mask = torch.tensor([True, False, True, False])
    def run():
        return scatter_visual_embeds_to_dense([[prefix, suffix]], mask, 4, 2,
            device="cpu", cache_owner=owner, cached_prefix_parts=1)[0]
    first = run()
    assert calls[-1] == [0, 2]
    assert run().data_ptr() == first.data_ptr() and calls[-1] == [2]
    prefix.add_(3)
    assert torch.equal(run()[0], prefix[0]) and calls[-1] == [0, 2]
    mask[0], mask[1] = False, True
    result = run()
    assert calls[-1] == [1, 2] and torch.equal(result[0], torch.zeros(2).half())
    assert torch.equal(result[1], prefix[0])


@pytest.mark.parametrize("parts", [False, True])
def test_padded_deepstack_refreshes_values_mask_and_execution_length(monkeypatch, parts):
    from types import SimpleNamespace as NS
    from rpu_backend.adapters.qwen3_vl.text import scatter_visual_embeds_to_dense

    def scatter(destination, indices, sources):
        destination.index_copy_(0, indices, torch.cat(sources))
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_scatter_row_parts_unflushed_", scatter, raising=False)
    owner = NS()
    source = torch.arange(4, dtype=torch.float16).view(2, 2)
    mask = torch.tensor([True, False, True, False])

    def run(execution_len, logical_len=4):
        embed = list(source.split(1)) if parts else source
        return scatter_visual_embeds_to_dense([embed], mask[:logical_len], logical_len, 2,
            device="cpu", cache_owner=owner, execution_len=execution_len)[0]

    first = run(6)
    assert first.shape == (6, 2) and not first[4:].any()
    source.add_(10)
    second = run(6)
    assert second.data_ptr() == first.data_ptr()
    assert torch.equal(second[[0, 2]], source) and not second[4:].any()
    mask[0], mask[1] = False, True
    changed_mask = run(6)
    assert changed_mask.data_ptr() == first.data_ptr()
    assert not changed_mask[[0, 3, 4, 5]].any()
    assert torch.equal(changed_mask[[1, 2]], source)
    larger = run(8)
    assert larger.shape == (8, 2) and not larger[4:].any()
    smaller = run(4)
    assert smaller.shape == (4, 2) and torch.equal(smaller[[1, 2]], source)
    # The same physical length can serve a shorter logical request; old rows
    # must not survive as a nonzero padding tail.
    source[1].fill_(9)
    previous = run(6)
    mask[2], mask[3] = False, True
    previous = run(6)
    assert previous[3].any()
    mask[0], mask[3] = True, False
    shorter = run(6, logical_len=3)
    assert torch.equal(shorter[[0, 1]], source) and not shorter[3:].any()


def test_padded_deepstack_rejects_short_execution_and_padding_indices(monkeypatch):
    from types import SimpleNamespace as NS
    from rpu_backend.adapters.qwen3_vl.text import scatter_visual_embeds_to_dense
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_scatter_row_parts_unflushed_",
        lambda *_: pytest.fail("invalid logical range reached native scatter"), raising=False)
    source = [torch.ones(1, 2).half()]
    mask = torch.tensor([True, False, False, False])
    with pytest.raises(ValueError, match="smaller than logical"):
        scatter_visual_embeds_to_dense(source, mask, 4, 2,
            device="cpu", cache_owner=NS(), execution_len=3)
    with pytest.raises(ValueError, match="logical seq_len"):
        scatter_visual_embeds_to_dense(source, mask, 4, 2, device="cpu",
            cache_owner=NS(), execution_len=6, row_indices_cpu=torch.tensor([4]))


@pytest.mark.parametrize("change_mask", [False, True])
def test_failed_deepstack_scatter_never_certifies_an_unwritten_prefix(monkeypatch, change_mask):
    from types import SimpleNamespace as NS
    from rpu_backend.adapters.qwen3_vl.text import scatter_visual_embeds_to_dense
    state = {"fail": False, "calls": []}
    def scatter(destination, indices, parts):
        state["calls"].append(indices.tolist())
        if state["fail"]:
            state["fail"] = False
            raise ValueError("native suffix prevalidation failed")
        destination.index_copy_(0, indices, torch.cat(parts))
        return destination
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_scatter_row_parts_unflushed_", scatter, raising=False)
    owner = NS()
    prefix, suffix = torch.ones(1, 2).half(), torch.full((1, 2), 2.).half()
    mask = torch.tensor([True, False, True, False])
    def run():
        return scatter_visual_embeds_to_dense([[prefix, suffix], [prefix, suffix]], mask, 4, 2,
            device="cpu", cache_owner=owner, cached_prefix_parts=1)
    run()
    if change_mask:
        mask[0], mask[1] = False, True
    else:
        prefix.add_(5)
    state["fail"] = True
    with pytest.raises(ValueError, match="prevalidation"):
        run()
    result = run()
    expected_rows = [1, 2] if change_mask else [0, 2]
    assert state["calls"][-2:] == [expected_rows, expected_rows]
    for dense in result:
        assert torch.equal(dense[expected_rows[0]], prefix[0])


def test_public_visual_scatter_retry_copies_prefix_after_rejected_suffix(monkeypatch):
    from types import SimpleNamespace as NS
    from rpu_backend.adapters import qwen3_vl as q
    from rpu_backend.api.cache import RPUCache
    original_to, original_empty = torch.Tensor.to, torch.empty
    def cpu_to(tensor, *args, **kwargs):
        if args and str(args[0]) == "rpu": args = ("cpu", *args[1:])
        if str(kwargs.get("device")) == "rpu": kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)
    def cpu_empty(*args, **kwargs):
        if str(kwargs.get("device")) == "rpu": kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch, "empty", cpu_empty)
    calls = []
    def gather(destination, weight, indices, skip):
        selected = indices != skip
        destination[selected] = weight[indices[selected]]
        return destination
    def scatter(destination, indices, parts):
        if any(part.ndim != 2 or part.shape[1] != destination.shape[1] for part in parts):
            raise ValueError("native suffix width rejected before writes")
        calls.append(indices.tolist())
        destination.index_copy_(0, indices, torch.cat(parts))
        return destination
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_gather_embedding_from_rpu_unflushed_", gather, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_scatter_row_parts_unflushed_", scatter, raising=False)
    embedding = torch.nn.Embedding(4, 2, dtype=torch.float16)
    text = NS()
    model = NS(model=NS(language_model=text, visual=NS()),
        config=NS(text_config=NS(hidden_size=2), vision_config=NS(spatial_merge_size=2), image_token_id=3),
        get_input_embeddings=lambda: embedding)
    cache = object.__new__(RPUCache)
    cache.position, cache.max_seq_len, cache.batch_size = 0, 64, 1
    prefix, suffix = torch.ones(1, 2).half(), torch.full((1, 2), 2.).half()
    def run(last):
        # Stop after the public scatter, before unrelated RoPE/model execution.
        return q._rpu_qwen3vl_forward.__wrapped__(model,
            input_ids=torch.tensor([[3, 0, 3, 0]]), past_key_values=cache,
            image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]),
            rpu_precomputed_vision={"pooler_output_parts": [prefix, last],
                "deepstack_feature_parts": [], "cached_prefix_parts": 1},
            cache_position=torch.tensor([-1]))
    with pytest.raises(NotImplementedError, match="cache_position disagrees"):
        run(suffix)
    prefix.add_(3)
    with pytest.raises(ValueError, match="suffix width"):
        run(torch.zeros(1, 3).half())
    assert model._rpu_visual_cached_prefix is None
    with pytest.raises(NotImplementedError, match="cache_position disagrees"):
        run(suffix)
    assert calls[-1] == [0, 2]
    slot, = text._rpu_qwen3vl_input_slots.values()
    assert torch.equal(slot[0, 0], prefix[0])


@pytest.mark.parametrize("kind,logical,execution", [
    ("decode", 1, 1),
    ("text", 3, 16),
    ("vision", 4, 4),
    ("vision", 4, 16),
    ("pixels_w8", 4, 4),
    ("pixels_fp16", 4, 4),
    ("pixels_eager", 4, 4),
    ("inputs_embeds", 1, 1),
])
@pytest.mark.parametrize("inference", [False, True])
def test_public_inputs_publish_completed_writes_before_decoder(
    monkeypatch, kind, logical, execution, inference,
):
    """Run the real producer chain; native producers publish before decoder consumption."""
    from contextlib import nullcontext
    from types import SimpleNamespace as NS
    from rpu_backend.adapters import qwen3_vl as q
    from rpu_backend.api.cache import RPUCache

    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if args and str(args[0]) == "rpu": args = ("cpu", *args[1:])
        if str(kwargs.get("device")) == "rpu": kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    for name in ("empty", "zeros"):
        original = getattr(torch, name)
        def allocate(*args, _original=original, **kwargs):
            if str(kwargs.get("device")) == "rpu": kwargs["device"] = "cpu"
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, allocate)

    def gather(destination, weight, indices, skip):
        assert indices.device.type == "cpu" and indices.is_contiguous()
        assert indices.data_ptr() == ids.data_ptr()
        selected = indices != skip
        destination[selected] = weight[indices[selected]]
        published[destination.data_ptr()] = destination.clone()
    def scatter(destination, indices, parts):
        destination.index_copy_(0, indices, torch.cat(parts))
        published[destination.data_ptr()] = destination.clone()
    published = {}
    def publish(tensors):
        pytest.fail("Consumer repeated publication already completed by native producers")
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_gather_embedding_from_rpu_unflushed_", gather, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_scatter_row_parts_unflushed_", scatter, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_publish_inputs", publish, raising=False)
    embedding = torch.nn.Embedding(4, 2, dtype=torch.float16)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(8).reshape(4, 2))
    text = NS(_rpu_decoder_handle=7,
              _rpu_text_graph_cache=NS(capture=lambda signature: nullcontext()))
    pixel_route = kind.startswith("pixels_")
    vision_calls, vision_owners = [], []
    class Vision:
        _rpu_vision_merger_on_device = kind != "pixels_eager"
        _rpu_vision_fused_merger_w8a16 = kind == "pixels_w8"
        _rpu_vision_chunked_merger_fp16 = kind == "pixels_fp16"
        def __call__(self, pixels, *, grid_thw, **kwargs):
            assert grid_thw.tolist() == [[1, 2, 2]]
            callback = kwargs.pop("_rpu_merged_consumer", None)
            assert (callback is not None) == (kind != "pixels_eager")
            assert kwargs == ({"_rpu_keep_merged_fp16": True} if kind != "pixels_eager" else {})
            pooler = pixels[:1].to(torch.float16 if kind != "pixels_eager" else torch.float32).clone()
            features = [pooler + i for i in range(3)]
            vision_owners.append((pooler, [*features]))
            vision_calls.append((callback is not None, pooler.dtype, pooler.device.type))
            # Actual top must consume only these fields, never raw hidden.
            if callback is not None:
                return callback(pooler, features)
            return NS(pooler_output=pooler, deepstack_features=features)
    model = NS(model=NS(language_model=text, visual=Vision(), rope_deltas=None),
        config=NS(text_config=NS(hidden_size=2, vocab_size=4),
                  vision_config=NS(spatial_merge_size=2), image_token_id=3),
        get_input_embeddings=lambda: embedding)
    cache = RPUCache(1, 1, 64, 1, 16, device="cpu")
    zero = torch.zeros(64, 2).half()
    monkeypatch.setattr(q, "make_zero_visual_embeds", lambda _, length: [zero[:length]] * 3)
    monkeypatch.setattr(q, "_qwen3_vl_prefill_plan", lambda *args: (execution, execution, NS()))
    monkeypatch.setattr(q, "_qwen3_vl_text_graph_signature", lambda *args: object())
    observed = []
    class ConsumerReached(Exception): pass
    def consume(owner, handle, **kwargs):
        hidden = kwargs["inputs_embeds"]
        dense = kwargs["deepstack_dense_visual_embeds"]
        positions = kwargs["position_ids"]
        assert positions.device.type == "cpu" and positions.dtype == torch.int32
        assert positions.shape == (execution, 3) and positions.is_contiguous()
        assert torch.equal(positions[:logical], supplied_positions.squeeze(1).T.to(torch.int32))
        if kind != "inputs_embeds":
            if execution == logical:
                assert torch.equal(published[hidden.data_ptr()].reshape_as(hidden), hidden)
            if kind == "vision" or pixel_route:
                assert all(torch.equal(published[value.data_ptr()], value) for value in dense)
        else:
            assert not published
        observed.append((hidden.clone(), [value.clone() for value in dense]))
        raise ConsumerReached
    monkeypatch.setattr(q, "_run_causal_decoder_forward", consume)

    owners = []
    ids = torch.zeros((1, logical), dtype=torch.long)
    ids_alias, ids_version = ids.numpy(), ids._version
    supplied_positions = torch.arange(logical).view(1, 1, logical).expand(3, 1, logical).clone()
    position_alias, position_version = supplied_positions.numpy(), supplied_positions._version
    pixels = torch.ones(4, 2)
    pixel_alias, pixel_version = pixels.numpy(), pixels._version
    for value in (0, 1, 0):
        # Same address/version, changed contents: a version-only IDs cache is invalid.
        ids_alias.fill(value)
        position_alias[:] = value + torch.arange(logical).numpy()
        args = dict(past_key_values=cache, position_ids=supplied_positions)
        if kind == "inputs_embeds":
            args["inputs_embeds"] = embedding(ids).detach()
        else:
            args["input_ids"] = ids
        if kind == "vision" or pixel_route:
            ids_alias[0, value] = 3  # move the image row without changing geometry
            args["image_grid_thw"] = torch.tensor([[1, 2, 2]])
            if pixel_route:
                pixel_alias.fill(10. + value)
                args["pixel_values"] = pixels
            else:
                feature = torch.full((1, 2), 10. + value).half()
                args["rpu_precomputed_vision"] = {"pooler_output_parts": [feature],
                    "deepstack_feature_parts": [[feature + i] for i in range(3)]}
        with torch.inference_mode(inference), pytest.raises(ConsumerReached):
            q._rpu_qwen3vl_forward.__wrapped__(model, **args)
        assert ids._version == ids_version
        assert supplied_positions._version == position_version
        owners.append(next(iter(text._rpu_qwen3vl_input_slots.values())))
    assert owners[0] is owners[1] is owners[2]
    assert torch.equal(observed[0][0], observed[2][0])
    assert not torch.equal(observed[0][0], observed[1][0])
    assert not observed[0][0][:, logical:].any()
    if kind == "vision" or pixel_route:
        for first, changed, restored in zip(observed[0][1], observed[1][1], observed[2][1]):
            assert torch.equal(first, restored) and not torch.equal(first, changed)
            assert not first[logical:].any()
    if pixel_route:
        assert pixels._version == pixel_version
        assert len(vision_calls) == 3
        dtype = torch.float32 if kind == "pixels_eager" else torch.float16
        assert vision_calls == [(kind != "pixels_eager", dtype, "cpu")] * 3
        assert torch.equal(vision_owners[0][0], vision_owners[2][0])
        assert not torch.equal(vision_owners[0][0], vision_owners[1][0])
    assert bool(published) is (kind != "inputs_embeds")


def test_standalone_cached_deepstack_uses_completed_producer_publications(monkeypatch):
    from types import SimpleNamespace as NS
    from rpu_backend.adapters.qwen3_vl.text import scatter_visual_embeds_to_dense
    original_zeros = torch.zeros
    def zeros(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)
    monkeypatch.setattr(torch, "zeros", zeros)
    events = []
    def scatter(destination, indices, parts):
        destination.index_copy_(0, indices, torch.cat(parts))
        assert torch.equal(destination[1], torch.tensor([3., 4.]).half())
        events.append("scatter-published")
    def publish(tensors):
        pytest.fail("DeepStack repeated the native producer publication")
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_scatter_row_parts_unflushed_", scatter, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_publish_inputs", publish, raising=False)
    result = scatter_visual_embeds_to_dense(
        [torch.tensor([[3., 4.]]).half()] * 3, torch.tensor([False, True]),
        2, 2, cache_owner=NS())
    assert len(result) == 3 and events == ["scatter-published"] * 3


@pytest.mark.parametrize("delivery", [False, True])
@pytest.mark.parametrize("source_kind", ["explicit", "decode", "vision"])
def test_position_transport_keeps_fixed_delivery_rpu_route(monkeypatch, delivery, source_kind):
    """Execute the real transport block, including all three existing slot paths."""
    import ast
    from types import SimpleNamespace as NS
    from rpu_backend.adapters import qwen3_vl as q
    tree = ast.parse(Path(q.__file__).read_text())
    forward = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "_rpu_qwen3vl_forward")
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets)
    start = next(i for i, node in enumerate(forward.body) if assigns(node, "attn_rpu"))
    end = next(i for i, node in enumerate(forward.body) if assigns(node, "rope_cos_il"))
    block = compile(ast.fix_missing_locations(ast.Module(body=forward.body[start:end],
                                                        type_ignores=[])), str(q.__file__), "exec")
    class RpuTensor(torch.Tensor):
        @property
        def device(self):
            return NS(type="rpu")
    uploads = []
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            uploads.append(tensor.clone())
            return tensor.clone().as_subclass(RpuTensor)
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    owner = NS()
    if delivery:
        owner._qwen3_vl_delivery_profile = q._QWEN3_VL_DELIVERY_PROFILE
    positions = torch.zeros(3, 1, 4, dtype=torch.long)
    prepared = torch.zeros(4, 3, dtype=torch.int32)
    positions_alias, prepared_alias = positions.numpy(), prepared.numpy()
    versions = positions._version, prepared._version
    outputs = []
    for value in (5, 9, 5):
        positions_alias.fill(value)
        prepared_alias.fill(value)
        namespace = dict(vars(q), text_model=owner, position_ids=positions,
                         logical_seq_len=4, execution_seq_len=4,
                         decode_position_slot_key=(0, (3, 1, 4)) if source_kind == "decode" else None,
                         decode_position_prepared={"position_ids_cpu_il": prepared},
                         decode_position_content_key=q._tensor_content_key(prepared),
                         prefill_position_cache={} if source_kind == "vision" else None)
        exec(block, namespace)
        actual = namespace["pos_rpu"]
        assert actual.dtype == torch.int32 and actual.shape == (4, 3) and actual.is_contiguous()
        assert actual.device.type == ("rpu" if delivery else "cpu")
        assert torch.all(actual.as_subclass(torch.Tensor) == value)
        assert (positions._version, prepared._version) == versions
        outputs.append((actual, actual.as_subclass(torch.Tensor).clone()))
    assert len(uploads) == (1 if delivery else 0)
    assert torch.equal(outputs[0][1], outputs[2][1]) and not torch.equal(outputs[0][1], outputs[1][1])
    if delivery:
        assert outputs[0][0] is outputs[1][0] is outputs[2][0]
    else:
        assert not any(name.endswith("_position_slots") for name in vars(owner))
