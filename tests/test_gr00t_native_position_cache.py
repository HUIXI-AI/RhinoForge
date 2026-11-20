"""GR00T native-position memoization through the actual Python backbone caller."""
import ast
from contextlib import nullcontext
import inspect
import os
import textwrap
from types import SimpleNamespace as NS

import pytest


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())


@pytest.mark.parametrize("first_inference", [False, True])
def test_denoise_cpu_masks_keep_exact_keys_and_stable_native_slot_inputs(monkeypatch, first_inference):
    """Actual Python caller; native/Graph device execution is a CPU double."""
    import ctypes
    import torch
    from rpu_backend.adapters.gr00t import runtime
    from test_graph_cache_freeze import _cache, _sig

    graph = _cache()
    descriptor = (19, 23)
    plan = NS(selected=NS(stage_tuple=NS(physical_descriptor=descriptor)),
              graph_key_words=lambda: (31,))
    owner = NS(_masks=None, _mask_key=None, _mask_attn_key=None,
               _img_tok=99, _d_handle=7, _d_gc=graph,
               _d_cache=NS(k_caches=[], v_caches=[]),
               _component_generations={runtime._ACTION_COMPONENT: 0},
               _record_component_plan=lambda *a, **kw: None)
    ids = torch.tensor([[10, 99, 20, 99]], dtype=torch.long)
    attention = torch.ones_like(ids)
    inputs = {"input_ids": ids, "attention_mask": attention,
              "state": torch.ones((1, 1, runtime._AD))}
    real_to = torch.Tensor.to
    def no_mask_upload(value, *args, **kwargs):
        assert not (args and str(args[0]) == "rpu")
        assert str(kwargs.get("device")) != "rpu"
        return real_to(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", no_mask_upload)
    monkeypatch.setattr(runtime, "_half_rpu", lambda value: value.half().contiguous())
    monkeypatch.setattr(runtime.rpu_backend.graph, "GraphSignature",
                        lambda **kw: _sig((kw["op_id"], *kw["shapes"], *kw["dyn_dims"])))
    monkeypatch.setattr(torch.ops.rpu, "gr00t_denoise_get_resolved_chunk_size", lambda h: 41, raising=False)
    calls, prepared = [], [None, None]
    def native(handle, noise, state, keys, values, vl, text, image, raw):
        assert handle == 7 and tuple(raw) == descriptor
        assert keys is owner._d_cache.k_caches and values is owner._d_cache.v_caches
        assert noise.dtype == state.dtype == torch.float16
        assert tuple(noise.shape) == (1, 40, 144) and tuple(state.shape) == (1, 1, 144)
        im = ids.eq(99).reshape(-1)
        valid = attention.eq(1).reshape(-1)
        for ordinal, (mask, subset) in enumerate(((text, ~im & valid), (image, im & valid))):
            assert mask.device.type == "cpu" and mask.dtype == torch.float16
            assert mask.is_contiguous() and tuple(mask.shape) == (41, 4)
            assert torch.equal(mask[0] == 0, subset)
            assert torch.isneginf(mask[:, ~subset]).all()
            if prepared[ordinal] is None:
                # This device double keeps ordinary slots; actual native mask
                # preparation/publication is covered by its C++ regression.
                with torch.inference_mode(False):
                    prepared[ordinal] = torch.empty((41, 16), dtype=torch.float16)
            padded = torch.nn.functional.pad(mask, (0, 12), value=-float("inf"))
            prepared[ordinal].copy_(padded)
        calls.append((text, image, *(slot.data_ptr() for slot in prepared)))
        score = int((prepared[0][0] == 0).long().dot(torch.arange(16)))
        score += 10 * int((prepared[1][0] == 0).long().dot(torch.arange(16)))
        return torch.full((1, 40, 144), float(score), dtype=torch.float16)
    monkeypatch.setattr(torch.ops.rpu, "gr00t_denoise_unroll_forward", native, raising=False)

    def run(inference):
        with torch.inference_mode(inference), torch.no_grad():
            return runtime.Gr00tN1d7VLA._denoise(owner, torch.zeros(1, 4, 8),
                                                inputs, 4, 0, (41, plan))
    first = run(first_inference)
    original_masks = owner._masks
    assert owner._masks[0] is calls[-1][0] and owner._masks[1] is calls[-1][1]
    assert torch.equal(first, run(not first_inference))
    assert owner._masks is original_masks
    assert first.dtype == torch.float32 and first.device.type == "cpu"
    assert tuple(first.shape) == (1, 40, 132) and first.stride() == (5760, 144, 1)

    # CPU aliases mutate both exact keys without changing a version counter.
    ids_version, mask_version = ids._version, attention._version
    ids_alias = (ctypes.c_int64 * ids.numel()).from_address(ids.data_ptr())
    mask_alias = (ctypes.c_int64 * attention.numel()).from_address(attention.data_ptr())
    ids_alias[0], mask_alias[2] = 99, 0
    changed = run(False)
    assert not torch.equal(first, changed) and owner._masks is not original_masks
    ids_alias[0], mask_alias[2] = 10, 1
    restored = run(True)
    assert torch.equal(first, restored) and torch.equal(first, run(False))
    assert ids._version == ids_version and attention._version == mask_version
    assert torch.equal(first, torch.full_like(first, 42))
    assert graph.size() == 1 and [replay for _, replay in graph._impl.touches] == [False, True, True, True, True]
    assert len({call[2:] for call in calls}) == 1 and calls[0][2] != calls[0][3]
    assert owner._masks[0] is not original_masks[0]


def _backbone_case(monkeypatch, *, enabled=True, padding=0, three_camera=False,
                   borrowed_vision=False):
    import torch
    from rpu_backend.adapters.gr00t import runtime
    from rpu_backend.api.cache import RPUCache

    hidden = 16 if three_camera else 2
    ids = torch.tensor([[10, 99, 99, 20]], dtype=torch.long)
    if three_camera:
        ids = torch.full((1, 225), 10, dtype=torch.long)
        for start in (10, 70, 130):
            ids[0, start:start + 49] = 99
    mask = torch.ones_like(ids)
    grid = torch.tensor([[1, 2, 4]], dtype=torch.long)
    inputs = {"input_ids": ids, "attention_mask": mask, "image_grid_thw": grid,
              "pixel_values": torch.tensor([1.0])}
    native_uploads, rope_calls, text_calls, vision_calls = [], [], [], []
    native_owners_during_upload = []
    assembly_calls, assembly_owners, vision_outputs = [], [], []
    padding_rows = [padding]
    bad_rope = [False]
    original_to, original_zeros = torch.Tensor.to, torch.zeros

    def cpu_to(tensor, *args, **kwargs):
        args = list(args)
        positional_rpu = bool(args) and str(args[0]) == "rpu"
        keyword_rpu = str(kwargs.get("device")) == "rpu"
        if positional_rpu or keyword_rpu:
            if tensor.dtype == torch.int32 and tuple(tensor.shape) == (ids.shape[1], 3):
                # Canonical layout and dtype must be prepared on CPU before upload.
                assert tensor.device.type == "cpu" and tensor.is_contiguous()
                assert owner._pos_native_ready is False
                native_owners_during_upload.append(owner._pos_native_rpu)
                native_uploads.append(tensor.clone())
            if positional_rpu:
                args[0] = "cpu"
            if keyword_rpu:
                kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    def cpu_zeros(*args, **kwargs):
        if str(kwargs.get("device")) == "rpu":
            kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)

    def rope_index(input_ids, *, image_grid_thw, attention_mask, **kwargs):
        # Each key affects values, making a stale hit observable, including
        # equal-shaped in-place input changes.
        base = input_ids[0] * 10 + attention_mask[0] * 100 + image_grid_thw.sum() * 1000
        value = torch.stack((base, base + 10000, base + 20000)).unsqueeze(1)
        if bad_rope[0]:
            value = value[:, :, :-1]
        rope_calls.append(value.clone())
        return value, None

    class Vision:
        _rpu_last_execution_plan = {"component": runtime._VISION_COMPONENT,
                                   "generation": 0, "dry_forward_agreement": True}

        def __call__(self, pixels, **kwargs):
            value = float(pixels.item())
            vision_calls.append(value)
            rows = 147 if three_camera else 2
            if not borrowed_vision or not hasattr(self, "slots"):
                with torch.inference_mode(False):
                    self.slots = [torch.empty((rows, hidden), dtype=torch.float16)
                                  for _ in range(4)]
            for slot, offset in zip(self.slots, (0, 10, 20, 30)):
                slot.fill_(value + offset)
            result = NS(pooler_output=self.slots[0], deepstack_features=self.slots[1:])
            vision_outputs.append(result)
            return kwargs["_rpu_merged_consumer"](
                result.pooler_output, result.deepstack_features)

    def assemble(prefix, assembly_ids, parts, runs, prefix_is_embeddings=False):
        import weakref
        # Device/transport double only: the real native guard and DMA methods
        # are compiled separately below. This consumer checks the actual caller.
        assert prefix_is_embeddings is True
        assert assembly_ids.dtype == torch.int32 and assembly_ids.is_contiguous()
        bank_count = 1 + len(vision_outputs[-1].deepstack_features)
        assert torch.equal(assembly_ids, inputs["input_ids"].reshape(-1).repeat(bank_count).int())
        assert runs.device.type == "cpu" and runs.dtype == torch.int64
        assert runs.is_contiguous() and tuple(runs.shape) == (len(parts), 2)
        assert tuple(prefix.shape) == (assembly_ids.numel(), hidden)
        output = prefix.clone()
        for part, (start, length) in zip(parts, runs.tolist()):
            assert part.is_contiguous() and part.dtype == torch.float16
            assert tuple(part.shape) == (length, hidden)
            output[start:start + length].copy_(part)
        assembly_calls.append((prefix, assembly_ids, runs, [part.data_ptr() for part in parts]))
        assembly_owners.append(weakref.ref(output))
        return output

    descriptor = (3, 7, 16)
    plan = NS(selected=NS(stage_tuple=NS(compute_chunk=16, physical_descriptor=descriptor)),
              graph_key_words=lambda: (1, 2))
    text = NS(_rpu_decoder_handle=7, _rpu_text_hidden_size=hidden, _rpu_text_num_layers=1,
              _rpu_text_deepstack_hash=0,
              _rpu_text_graph_cache=NS(capture=lambda signature: nullcontext()))
    cache = RPUCache(1, 1, 256 if three_camera else 64, 1, 16, device="cpu")
    owner = NS(_qwen_z1_plan_hash=None,
        _bb=NS(config=NS(image_token_id=99), model=NS(get_rope_index=rope_index),
               get_input_embeddings=lambda: lambda values: values.unsqueeze(-1).expand(-1, -1, hidden).half().clone()),
        _cfg=NS(text_config=NS(hidden_size=hidden)), _text=text, _vision=Vision(),
        _component_generations={runtime._VISION_COMPONENT: 0, runtime._TEXT_COMPONENT: 0},
        _rpu_last_execution_plan={}, _emb_text_rpu=None, _emb_key=None, _img_runs=None,
        _ds_dense=None, _rope_ids=None, _rope_key=None, _rope_grid_key=None, _rope_attn_key=None,
        _pos_rpu=None, _pos_native_rpu=None, _pos_native_ready=False, _attn_rpu=None, _partial_mrope=False,
        _native_position_cache=enabled, _rope_cos_il=None, _rope_sin_il=None, _bb_cache=cache,
        _prefill_execution_plan=lambda length: (length + padding_rows[0], 16, plan),
        _record_component_plan=lambda *args, **kwargs: {"authority": "NATIVE_A6_STAGE_DESCRIPTOR", **kwargs})

    def shared_forward(model, handle, **kwargs):
        position = kwargs["position_ids"]
        text_calls.append(NS(position=position, position_values=position.clone(),
            embeds=kwargs["inputs_embeds"].clone(),
            dense=[value.clone() for value in kwargs["deepstack_dense_visual_embeds"]]))
        execution_len = int(kwargs["inputs_embeds"].shape[1])
        model._rpu_last_execution_plan = {"logical_len": ids.shape[1], "execution_len": execution_len,
            "chunk_size": 16, "position": 0, "physical_descriptor": descriptor}
        cache.reset_to_position(execution_len)
        return kwargs["inputs_embeds"].clone(), cache

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(torch.ops.rpu, "assemble_inputs_embeds_parts", assemble, raising=False)
    monkeypatch.setattr(runtime, "_run_causal_decoder_forward", shared_forward)
    monkeypatch.setattr(runtime, "chunk_policy_key", lambda handle: 0)
    return NS(runtime=runtime, owner=owner, inputs=inputs, native_uploads=native_uploads,
              rope_calls=rope_calls, text_calls=text_calls, vision_calls=vision_calls,
              padding_rows=padding_rows, bad_rope=bad_rope, assembly_calls=assembly_calls,
              assembly_owners=assembly_owners, vision_outputs=vision_outputs,
              native_owners_during_upload=native_owners_during_upload,
              run=lambda: runtime.Gr00tN1d7VLA._backbone(owner, inputs))


def test_native_position_cache_reuses_prompt_but_refreshes_images(monkeypatch):
    import torch
    case = _backbone_case(monkeypatch)
    snapshots = []
    # Same tensor owner and prompt throughout; image contents change in place.
    for image in (1.0, 7.0, 7.0, 1.0):
        case.inputs["pixel_values"].fill_(image)
        raw, length = case.run()
        assert raw.shape == (1, 4, 2) and length == 4
        call = case.text_calls[-1]
        assert call.position.dtype == torch.int32 and call.position.is_contiguous()
        assert torch.equal(call.position_values, case.rope_calls[0].squeeze(1).T.contiguous().int())
        assert torch.equal(call.embeds[0, 1:3], torch.full((2, 2), image, dtype=torch.float16))
        assert torch.equal(call.embeds[0, [0, 3]], torch.tensor([[10, 10], [20, 20]]).half())
        for dense, offset in zip(call.dense, (10, 20, 30)):
            assert torch.equal(dense[1:3], torch.full((2, 2), image + offset, dtype=torch.float16))
            assert torch.count_nonzero(dense[[0, 3]]) == 0
        snapshots.append(call.embeds)
    assert len(case.rope_calls) == len(case.native_uploads) == 1
    assert case.vision_calls == [1.0, 7.0, 7.0, 1.0]
    assert all(call.position is case.owner._pos_native_rpu for call in case.text_calls)
    assert len({call.position.data_ptr() for call in case.text_calls}) == 1
    assert torch.equal(snapshots[1], snapshots[2]) and torch.equal(snapshots[0], snapshots[3])
    assert not torch.equal(snapshots[0], snapshots[1])
    assert len(case.assembly_calls) == 4
    for prefix, assembly_ids, runs, _parts in case.assembly_calls:
        assert prefix is case.owner._emb_prefix_rpu
        assert tuple(prefix.shape) == (16, 2) and assembly_ids.numel() == 16
        assert runs.tolist() == [[1, 2], [5, 2], [9, 2], [13, 2]]


def test_native_position_cache_refreshes_on_inplace_prompt_grid_mask_changes(monkeypatch):
    import torch
    case = _backbone_case(monkeypatch)
    case.run()
    for tensor, index in ((case.inputs["input_ids"], (0, 0)),
                          (case.inputs["image_grid_thw"], (0, 1)),
                          (case.inputs["attention_mask"], (0, 3))):
        previous = case.owner._pos_native_rpu
        old_values = previous.clone()
        tensor[index] += 1
        case.run()
        current = case.owner._pos_native_rpu
        assert current is not previous and current.data_ptr() != previous.data_ptr()
        assert case.native_owners_during_upload[-1] is previous
        assert case.owner._pos_native_ready is True
        assert not torch.equal(current, old_values)
        assert torch.equal(current, case.rope_calls[-1].squeeze(1).T.contiguous().int())
        assert case.text_calls[-1].position is current
        case.run()
        assert case.owner._pos_native_rpu is current
    assert len(case.rope_calls) == len(case.native_uploads) == 4
    # A malformed new RoPE result must never reach native text dispatch.
    previous_calls = len(case.text_calls)
    case.bad_rope[0] = True
    case.inputs["input_ids"][0, 0] += 1
    previous_uploads = len(case.native_uploads)
    previous_owner = case.owner._pos_native_rpu
    for _ in range(2):
        with pytest.raises((ValueError, RuntimeError), match="position|RoPE|rope"):
            case.run()
        assert len(case.text_calls) == previous_calls
        assert case.owner._pos_native_rpu is previous_owner
        assert case.owner._pos_native_ready is False
        assert len(case.native_uploads) == previous_uploads
    # Keys were already copied during the failed attempts; the not-ready flag
    # must force a retry while retaining the prior owner through allocation.
    case.bad_rope[0] = False
    case.run()
    assert len(case.text_calls) == previous_calls + 1
    assert len(case.native_uploads) == previous_uploads + 1
    assert case.native_owners_during_upload[-1] is previous_owner
    assert case.owner._pos_native_rpu is not previous_owner
    assert case.owner._pos_native_ready is True
    assert torch.equal(case.text_calls[-1].position, case.rope_calls[-1].squeeze(1).T.contiguous().int())


@pytest.mark.parametrize("mode", ("off", "padding", "z1"))
def test_native_position_cache_preserves_off_padding_and_z1_paths(monkeypatch, mode):
    import torch
    case = _backbone_case(monkeypatch, enabled=mode != "off", padding=4 if mode == "padding" else 0)
    if mode == "z1":
        expected = object()
        case.owner._qwen_z1_plan_hash = 123
        case.owner._backbone_qwen_z1 = lambda inputs: expected
        assert case.run() is expected
        assert not case.vision_calls and not case.text_calls and not case.native_uploads
        return
    case.run()
    position = case.text_calls[-1].position
    assert position.shape == (3, 1, 8 if mode == "padding" else 4)
    assert position.dtype == torch.int64
    assert torch.equal(position[:, :, :4], case.rope_calls[-1])
    if mode == "off":
        assert not case.native_uploads and case.owner._pos_native_rpu is None
    else:
        assert torch.equal(position[:, :, 4:], position[:, :, 3:4].expand(3, 1, 4))
        assert len(case.native_uploads) == 1 and position is not case.owner._pos_native_rpu
        # Returning to no padding selects the already-owned canonical table.
        case.padding_rows[0] = 0
        case.run()
        assert case.text_calls[-1].position is case.owner._pos_native_rpu
        assert len(case.native_uploads) == 1


def test_native_position_cache_reader_and_constructor_are_cold(monkeypatch):
    from rpu_backend.adapters.gr00t import runtime
    name = "RPU_GR00T_NATIVE_POSITION_CACHE"
    monkeypatch.delenv(name, raising=False)
    assert runtime._native_position_cache_on() is False
    for value, expected in (("", False), ("0", False), ("1", True)):
        monkeypatch.setenv(name, value)
        assert runtime._native_position_cache_on() is expected
    for value in ("true", "false", "on", "01", "2", "-1", " 1", "1 "):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match=name):
            runtime._native_position_cache_on()
    # Execute only the actual constructor assignment, avoiding a second copy
    # of the owner's device installation and retirement lifecycle.
    constructor = ast.parse(textwrap.dedent(inspect.getsource(runtime.Gr00tN1d7VLA.__init__)))
    assignments = [node for node in ast.walk(constructor) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Attribute) and target.attr == "_native_position_cache"
                           for target in node.targets)]
    assert len(assignments) == 1
    code = compile(ast.fix_missing_locations(ast.Module(body=assignments, type_ignores=[])), "<cold-cache-assignment>", "exec")
    monkeypatch.setenv(name, "1")
    reads = []
    def read():
        reads.append(1)
        return runtime._native_position_cache_on()
    owner = NS()
    exec(code, {"self": owner, "qwen_z1_plan_hash": None, "_native_position_cache_on": read})
    assert owner._native_position_cache is True and len(reads) == 1
    monkeypatch.setenv(name, "0")
    assert owner._native_position_cache is True and len(reads) == 1
    z1 = NS()
    exec(code, {"self": z1, "qwen_z1_plan_hash": 123, "_native_position_cache_on": read})
    assert z1._native_position_cache is False and len(reads) == 1


def test_backbone_without_image_runs_preserves_cold_logical_banks(monkeypatch):
    import torch
    case = _backbone_case(monkeypatch)
    case.inputs["input_ids"].fill_(10)
    retained = []
    for image in (1.0, 7.0, 1.0):
        case.inputs["pixel_values"].fill_(image)
        result, length = case.run()
        retained.append(result)
        assert length == 4 and torch.equal(result, torch.full((1, 4, 2), 10.0).half())
        assert all(torch.count_nonzero(value) == 0 for value in case.text_calls[-1].dense)
    assert not case.assembly_calls and not case.assembly_owners
    assert len(case.rope_calls) == 1 and torch.equal(retained[0], retained[2])


def test_backbone_combined_parts_preserves_native_shape_rejection_before_text(monkeypatch):
    case = _backbone_case(monkeypatch, three_camera=True)
    original_vision = case.owner._vision

    def short_deepstack(*args, **kwargs):
        consume = kwargs["_rpu_merged_consumer"]
        kwargs["_rpu_merged_consumer"] = lambda merged, ds: consume(
            merged, [*ds[:-1], ds[-1][:-1]])
        return original_vision(*args, **kwargs)

    short_deepstack._rpu_last_execution_plan = original_vision._rpu_last_execution_plan
    case.owner._vision = short_deepstack
    # The existing native admission is exercised by the compiled node below;
    # its shape-checking transport double verifies this caller passes every bank.
    with pytest.raises(AssertionError):
        case.run()
    assert not case.text_calls and not case.assembly_calls
    case.owner._vision = original_vision
    case.run()
    assert len(case.text_calls) == len(case.assembly_calls) == 1


@pytest.mark.parametrize("inference", (False, True))
@pytest.mark.parametrize("borrowed_vision", (False, True))
def test_backbone_parts_owners_survive_deferred_capture_and_raw_alias_aba(
        monkeypatch, inference, borrowed_vision):
    import ctypes
    import gc
    import weakref

    import torch
    from test_graph_cache_freeze import _FakeGraph, _cache, _sig

    case = _backbone_case(monkeypatch, three_camera=True, borrowed_vision=borrowed_vision)
    cache = _cache()
    case.owner._text._rpu_text_graph_cache = cache
    monkeypatch.setattr(case.runtime.rpu_backend.graph, "GraphSignature",
                        lambda **kwargs: _sig((tuple(kwargs["shapes"]), tuple(kwargs["dyn_dims"]))))
    pending, exits, consumers = [], [], []
    descriptor = (3, 7, 16)

    def consume(model, handle, **kwargs):
        inputs = kwargs["inputs_embeds"]
        dense = kwargs["deepstack_dense_visual_embeds"]
        assert tuple(inputs.shape) == (1, 225, 16) and len(dense) == 3
        assert all(tuple(value.shape) == (225, 16) for value in dense)
        banks = [inputs, *dense]
        assert all(value.is_contiguous() for value in banks)
        assert len({value.untyped_storage().data_ptr() for value in banks}) == 1
        assert [value.storage_offset() for value in banks] == [bank * 225 * 16 for bank in range(4)]
        assert inputs.untyped_storage().nbytes() == 4 * 225 * 16 * 2
        output = torch.empty_like(inputs)
        # Mutable native bindings refresh on each call. They are deliberately
        # weak here, so capture.end() must consume live local owners.
        pending[:] = [weakref.ref(inputs), *(weakref.ref(value) for value in dense), weakref.ref(output)]
        model._rpu_last_execution_plan = {"logical_len": 225, "execution_len": 225,
            "chunk_size": 16, "position": 0, "physical_descriptor": descriptor}
        case.owner._bb_cache.reset_to_position(225)
        return output, case.owner._bb_cache

    original_end = _FakeGraph.end

    def end(graph):
        gc.collect()
        inputs, *rest = [reference() for reference in pending]
        assert inputs is not None and all(value is not None for value in rest)
        dense, output = rest[:3], rest[3]
        whole_owner = case.assembly_owners[-1]()
        assert whole_owner is not None and tuple(whole_owner.shape) == (900, 16)
        assert whole_owner.data_ptr() == inputs.data_ptr()
        consumers.append((inputs.clone(), [value.clone() for value in dense]))
        output.copy_(inputs + sum(dense).unsqueeze(0))
        exits.append(graph.state())
        pending.clear()
        original_end(graph)

    monkeypatch.setattr(case.runtime, "_run_causal_decoder_forward", consume)
    monkeypatch.setattr(_FakeGraph, "end", end)
    pixel = case.inputs["pixel_values"]
    raw_pixel = torch.frombuffer((ctypes.c_ubyte * pixel.nbytes).from_address(pixel.data_ptr()), dtype=pixel.dtype)
    ids = case.inputs["input_ids"]
    raw_ids = torch.frombuffer((ctypes.c_ubyte * ids.nbytes).from_address(ids.data_ptr()), dtype=ids.dtype).reshape_as(ids)
    versions = pixel._version, ids._version
    retained, skeletons, prefixes, vision_snapshots = [], [], [], []
    for value in (1.0, 7.0, 1.0):
        raw_pixel.fill_(value)
        with torch.inference_mode(inference), torch.no_grad():
            result, length = case.run()
        assert length == 225 and result.shape == (1, 225, 16)
        inputs, dense = consumers[-1]
        oracle = ids.unsqueeze(-1).expand(1, 225, 16).half().clone()
        dense_oracle = [torch.zeros(225, 16, dtype=torch.float16) for _ in range(3)]
        for start in (10, 70, 130):
            oracle[0, start:start + 49] = value
            for target, offset in zip(dense_oracle, (10, 20, 30)):
                target[start:start + 49] = value + offset
        assert torch.equal(inputs, oracle)
        assert all(torch.equal(actual, expected) for actual, expected in zip(dense, dense_oracle))
        assert torch.equal(result, oracle + sum(dense_oracle).unsqueeze(0))
        retained.append(result)
        skeletons.append(case.owner._emb_text_rpu)
        prefixes.append(case.owner._emb_prefix_rpu)
        vision = case.vision_outputs[-1]
        vision_snapshots.append((vision.pooler_output.clone(), [v.clone() for v in vision.deepstack_features]))
        gc.collect()
        # Fresh output is retained by the caller; assembled input banks are
        # request-local and no longer held after the synchronous graph exit.
        assert all(reference() is None for reference in case.assembly_owners)
    assert pixel._version == versions[0] and ids._version == versions[1]
    assert exits == [1, 3, 3] and cache.size() == 1
    assert cache._impl.touches == [(cache._impl.touches[0][0], False),
                                   (cache._impl.touches[0][0], True),
                                   (cache._impl.touches[0][0], True)]
    assert skeletons[0] is skeletons[1] is skeletons[2]
    assert prefixes[0] is prefixes[1] is prefixes[2]
    assert len(case.assembly_calls) == 3
    expected_runs = [[bank * 225 + start, 49]
                     for bank in range(4) for start in (10, 70, 130)]
    assert all(call[2].tolist() == expected_runs for call in case.assembly_calls)
    assert all(len(call[3]) == 12 for call in case.assembly_calls)
    assert torch.equal(skeletons[0], ids.unsqueeze(-1).expand(1, 225, 16).half())
    assert torch.equal(prefixes[0][:225], skeletons[0][0])
    assert torch.count_nonzero(prefixes[0][225:]) == 0
    assert all(torch.count_nonzero(value) == 0 for value in case.owner._ds_dense)
    assert all(not value.is_inference() for value in [skeletons[0], prefixes[0], case.owner._emb_assembly_ids,
                                                     case.owner._img_runs_cpu, *case.owner._ds_dense])
    assert torch.equal(retained[0], retained[2]) and not torch.equal(retained[0], retained[1])
    if borrowed_vision:
        assert len({value.pooler_output.data_ptr() for value in case.vision_outputs}) == 1
        assert not torch.equal(case.vision_outputs[1].pooler_output, vision_snapshots[1][0])
        assert len({tuple(call[3]) for call in case.assembly_calls}) == 1
    else:
        for visual, (pooler, deepstack) in zip(case.vision_outputs, vision_snapshots):
            assert torch.equal(visual.pooler_output, pooler)
            assert all(torch.equal(actual, saved) for actual, saved in zip(visual.deepstack_features, deepstack))
    # Changing both real text IDs and the image runs without a version bump
    # refreshes the immutable cache atomically. The old snapshot stays intact.
    old_ids, old_runs = case.owner._emb_assembly_ids, case.owner._img_runs_cpu
    old_ids_values, old_runs_values = old_ids.clone(), old_runs.clone()
    raw_ids[0, 0] = 20
    raw_ids[0, 10] = 10
    raw_ids[0, 59] = 99
    with torch.inference_mode(inference), torch.no_grad():
        result, _ = case.run()
    assert ids._version == versions[1]
    assert case.owner._emb_text_rpu is not skeletons[0]
    assert case.owner._emb_prefix_rpu is not prefixes[0]
    assert case.owner._emb_assembly_ids is not old_ids and case.owner._img_runs_cpu is not old_runs
    assert torch.equal(old_ids, old_ids_values) and torch.equal(old_runs, old_runs_values)
    assert case.owner._img_runs_cpu.tolist() == [
        [bank * 225 + start, 49] for bank in range(4) for start in (11, 70, 130)]
    assert len(case.assembly_calls) == 4
    assert result[0, 0, 0] == 20 and result[0, 10, 0] == 10
    assert result[0, 59, 0] == 1 + 11 + 21 + 31
    assert exits == [1, 3, 3, 3] and cache.size() == 1
    assert torch.equal(retained[0], retained[2])


def test_backbone_parts_actual_native_dma_publication_and_spm_admission(tmp_path, _extended_checks=False):
    """Compile actual input_builder, DMA guards and batch submission; no SDK claim.

    ATen backing/views are real; device labels, malformed metadata controls,
    allocation registry and synchronous queue transport are doubled.
    Half storage remains two bytes and separate CPU/device byte stores make
    unpublished aliases observable under all normal/force combinations.
    """
    from pathlib import Path
    from test_embedding_parts import _run_preassembled_parts_native_fixture
    from test_siglip_attention_dry_live_contract import _function

    root = Path(__file__).resolve().parents[1]
    input_builder = _function((root / "src/ops/rpu_embedding.cpp").read_text(),
                          "at::Tensor rpu_assemble_inputs_embeds_parts(")
    source = (root / "src/ops/rpu_memcpy.cpp").read_text()
    dma = "\n".join(_function(source, marker) for marker in (
        "void rpu_launch_ddr_broadcast_spm_dma_immediate(",
        "void rpu_launch_spm_copy_ddr_dma_immediate("))
    profile = (root / "src/core/rpu_profile.h").read_text()
    macros = "\n".join(profile[profile.index(f"#define {name}("):].split("\n\n", 1)[0]
                       for name in ("rpu_ddr_flush", "rpu_ddr_flush_force"))
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#define RECORD_FUNCTION(...) ((void)0)
#define TORCH_CHECK(c, ...) do {if(!(c))throw std::runtime_error(#c);}while(false)
namespace c10 {using Half=uint16_t;}
bool g_rpu_ddr_flush_enabled=false,g_rpu_ddr_flush_force_enabled=true;
int allocations=0,dma_count=0,submits=0,fail_submit=0;
struct Memory {
 size_t bytes;std::shared_ptr<uint8_t>cpu;std::vector<uint8_t>device;bool device_dirty=false;
 explicit Memory(size_t n):bytes(n),cpu(static_cast<uint8_t*>(std::aligned_alloc(256,(n+255)/256*256)),std::free),device(n){
  assert(cpu);std::memset(cpu.get(),0,n);++allocations;
 }
};
std::vector<std::shared_ptr<Memory>>memories;
std::pair<Memory*,size_t>lookup(uint64_t address){
 for(auto&m:memories){auto p=reinterpret_cast<uint64_t>(m->cpu.get());if(address>=p&&address-p<m->bytes)return{m.get(),address-p};}
 throw std::runtime_error("unowned address");
}
namespace rhino_lkn {
void RpuDdrFlush(void*p,size_t bytes){auto[m,o]=lookup(reinterpret_cast<uint64_t>(p));assert(bytes<=m->bytes-o);std::memcpy(m->device.data()+o,m->cpu.get()+o,bytes);}
void RpuDdrFlush(void*p){auto[m,o]=lookup(reinterpret_cast<uint64_t>(p));
 if(m->device_dirty){std::memcpy(m->cpu.get()+o,m->device.data()+o,m->bytes-o);m->device_dirty=false;}
 else std::memcpy(m->device.data()+o,m->cpu.get()+o,m->bytes-o);
}
}
uint64_t RpuGetDevAddr(const void*p){return reinterpret_cast<uint64_t>(p);}
namespace at {
enum ScalarType{kHalf,kInt,kLong};enum DeviceType{kCPU,kPrivateUse1};
struct Device{DeviceType value=kPrivateUse1;DeviceType type()const{return value;}bool is_cpu()const{return value==kCPU;}bool operator==(Device o)const{return value==o.value;}};
struct Options{ScalarType dtype;Device device;};
struct Tensor{
 std::shared_ptr<Memory>m;std::vector<int64_t>sizes;ScalarType dtype=kHalf;Device dev;size_t offset=0;bool contiguous=true,grad=false;
 Tensor()=default;
 Tensor(std::vector<int64_t>s,ScalarType d=kHalf,Device v={}):sizes(s),dtype(d),dev(v){
  size_t count=1;for(auto x:s)count*=x;m=std::make_shared<Memory>(count*(d==kHalf?2:d==kInt?4:8));memories.push_back(m);
 }
 int dim()const{return sizes.size();}int64_t size(int n)const{return sizes.at(n);}ScalarType scalar_type()const{return dtype;}
 Device device()const{return dev;}bool is_contiguous()const{return contiguous;}bool requires_grad()const{return grad;}
 void*data_ptr()const{return m->cpu.get()+offset;}template<class T>T*data_ptr()const{return reinterpret_cast<T*>(data_ptr());}
 Options options()const{return{dtype,dev};}
};
using TensorList=std::vector<Tensor>;
Tensor empty(std::initializer_list<int64_t>s,Options o){return Tensor(s,o.dtype,o.device);}
}
std::vector<uint8_t>spm(8<<20);
struct Node{uint64_t src,dst;size_t bytes;};
struct Queue_t{std::vector<Node>nodes;void add_barrier(uint8_t,uint8_t){}};
struct QueueCache{Queue_t queue;static QueueCache&instance(){static QueueCache q;return q;}Queue_t*get(uint8_t n){assert(n==1);return &queue;}};
constexpr uint64_t SPM_BASE=0x10000000;
uint64_t spm_unified_to_per_core_abs(int core,uint32_t a){assert(core==0);return a;}
void rpu_add_dma_checked(Queue_t&q,uint64_t s,uint64_t d,size_t bytes,int core,const char*){assert(core==0);q.nodes.push_back({s,d,bytes});++dma_count;}
void submit_immediate_batch(Queue_t*q,const char*){
 auto nodes=std::move(q->nodes);q->nodes.clear();++submits;if(fail_submit==submits)throw std::runtime_error("injected submit failure");
 for(auto n:nodes){
  if(n.dst>=SPM_BASE&&n.dst-SPM_BASE<spm.size()){auto[m,o]=lookup(n.src);assert(n.bytes<=m->bytes-o&&n.bytes<=spm.size()-(n.dst-SPM_BASE));std::memcpy(spm.data()+n.dst-SPM_BASE,m->device.data()+o,n.bytes);}
  else{assert(n.src>=SPM_BASE&&n.bytes<=spm.size()-(n.src-SPM_BASE));auto[m,o]=lookup(n.dst);assert(n.bytes<=m->bytes-o);std::memcpy(m->device.data()+o,spm.data()+n.src-SPM_BASE,n.bytes);m->device_dirty=true;}
 }
}
struct RpuKernelGraph{static bool active;static void check_direct_immediate_dma_allowed(const char*){TORCH_CHECK(!active);}};bool RpuKernelGraph::active=false;
namespace graph_dma{bool active(){return RpuKernelGraph::active;}}
struct ScopedEagerAtomicBatch{ScopedEagerAtomicBatch(bool,int){}void submit(){}};
struct SpmAllocator{
 static constexpr size_t SPM_PLANNING_BUDGET=8303708,ALIGN=256;
 struct AllocRequest{int64_t bytes;int first,last;};
 static bool scratch;static size_t available;static int releases;bool initialized=true;
 struct ScopedTemporary{~ScopedTemporary(){if(scratch){scratch=false;++releases;}}};
 bool is_initialized()const{return initialized;}void init(){initialized=true;}size_t free_space()const{return available;}
 std::vector<uint32_t>alloc_temporary_aliased(std::initializer_list<AllocRequest>requests){assert(!scratch&&requests.size()==1);auto r=*requests.begin();assert(r.bytes==921600&&size_t(r.bytes)<=available);scratch=true;return{0};}
 uint32_t addr(int core,uint32_t off){assert(core==0&&off==0&&scratch);return SPM_BASE;}
}SPM_ALLOC;
bool SpmAllocator::scratch=false;size_t SpmAllocator::available=SpmAllocator::SPM_PLANNING_BUDGET;int SpmAllocator::releases=0;
void rpu_launch_gather_embedding_kernel(c10::Half*,int32_t*,uint32_t,int64_t,int64_t,int){throw std::runtime_error("prefix route must not gather");}
void rpu_launch_ddr_broadcast_spm_dma_immediate(c10::Half*,int64_t,uint32_t,int);
void rpu_launch_spm_copy_ddr_dma_immediate(uint32_t,c10::Half*,int64_t);
void rpu_launch_ddr_broadcast_spm_dma(c10::Half*s,int64_t n,uint32_t d,int c){rpu_launch_ddr_broadcast_spm_dma_immediate(s,n,d,c);}
void rpu_launch_spm_copy_ddr_dma(uint32_t s,c10::Half*d,int64_t n){rpu_launch_spm_copy_ddr_dma_immediate(s,d,n);}
''' + macros + '\n' + dma + '\n' + input_builder + r'''
int main(){
 constexpr int64_t N=225,H=2048,L=49;constexpr size_t BYTES=N*H*2;
 at::Tensor prefix({N,H}),ids({N},at::kInt),runs({3,2},at::kLong,{at::kCPU}),features({3*L,H});
 auto run=runs.data_ptr<int64_t>();run[0]=10;run[1]=L;run[2]=70;run[3]=L;run[4]=130;run[5]=L;
 at::TensorList parts;for(int i=0;i<3;++i){auto part=features;part.sizes={L,H};part.offset=i*L*H*2;parts.push_back(part);}
 auto invoke=[&](const at::Tensor&p,const at::Tensor&i,const at::TensorList&f,const at::Tensor&r){return rpu_assemble_inputs_embeds_parts(p,i,f,r,true);};
 // Actual fp16 geometry and CPU-dirty alias A/B/A. Force publication models
 // the existing producer upload; normal=true additionally publishes DMA inputs.
 for(bool normal:{false,true})for(bool force:{false,true}){
  g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;
  std::vector<at::Tensor>retained;std::vector<std::vector<uint8_t>>saved;
  for(uint16_t value:{uint16_t(1),uint16_t(7),uint16_t(1)}){
   for(int component=0;component<4;++component){
    std::fill_n(prefix.data_ptr<uint16_t>(),N*H,component?0:11);
    std::fill_n(features.data_ptr<uint16_t>(),3*L*H,uint16_t(value+component*10));
    rpu_ddr_flush_force(prefix.data_ptr());rpu_ddr_flush_force(features.data_ptr());
    // Expected device bytes are from the last actual publication, rather than
    // pretending disabled flags make fresh CPU aliases visible.
    std::vector<uint8_t>expected(prefix.m->device);
    if(normal)std::memcpy(expected.data(),prefix.data_ptr(),BYTES);
    for(int camera=0;camera<3;++camera){
     const auto*src=normal?static_cast<uint8_t*>(parts[camera].data_ptr()):features.m->device.data()+parts[camera].offset;
     std::memcpy(expected.data()+run[2*camera]*H*2,src,L*H*2);
    }
    auto output=invoke(prefix,ids,parts,runs);
    assert(output.m!=prefix.m&&output.m!=features.m&&output.m->device==expected);
    if(normal||force){assert(reinterpret_cast<uint16_t*>(output.m->device.data())[10*H]==value+component*10);}
    else assert(reinterpret_cast<uint16_t*>(output.m->device.data())[10*H]!=value+component*10);
    for(size_t j=0;j<retained.size();++j)assert(retained[j].m->device==saved[j]);
    retained.push_back(output);saved.push_back(expected);assert(!SpmAllocator::scratch);
   }
  }
  // Device source content from the previous flag arm must not leak into the
  // deliberately unpublished controls in the next independent flag arm.
  std::fill(prefix.m->device.begin(),prefix.m->device.end(),0);std::fill(features.m->device.begin(),features.m->device.end(),0);
 }
 // All malformed input/current-SPM/current-Graph cases fail before output
 // allocation or the first queued DMA, including overlap and short sources.
 auto reject=[&](at::Tensor p,at::Tensor i,at::TensorList f,at::Tensor r){
  int a=allocations,d=dma_count,s=submits;bool rejected=false;
  try{invoke(p,i,f,r);}catch(const std::runtime_error&){rejected=true;}
  assert(rejected&&allocations==a&&dma_count==d&&submits==s&&!SpmAllocator::scratch);
 };
 {auto p=prefix;p.contiguous=false;reject(p,ids,parts,runs);}
 {auto p=prefix;p.dev={at::kCPU};reject(p,ids,parts,runs);}
 {auto p=prefix;p.dtype=at::kInt;reject(p,ids,parts,runs);}
 {auto p=prefix;p.grad=true;reject(p,ids,parts,runs);}
 {auto p=prefix;p.sizes[0]=224;reject(p,ids,parts,runs);}
 {auto p=prefix;p.sizes[1]=2047;reject(p,ids,parts,runs);}
 {auto p=prefix;p.sizes[1]=65536;reject(p,ids,parts,runs);}
 {auto p=prefix;p.offset=2;reject(p,ids,parts,runs);}
 {auto i=ids;i.dtype=at::kLong;reject(prefix,i,parts,runs);}
 {auto f=parts;f[0].sizes[0]=48;reject(prefix,ids,f,runs);}
 {auto f=parts;f[0].offset+=2;reject(prefix,ids,f,runs);}
 {auto f=parts;f[0].contiguous=false;reject(prefix,ids,f,runs);}
 {auto f=parts;f[0].grad=true;reject(prefix,ids,f,runs);}
 run[0]=-1;reject(prefix,ids,parts,runs);run[0]=10;
 run[2]=20;reject(prefix,ids,parts,runs);run[2]=70;
 run[4]=200;reject(prefix,ids,parts,runs);run[4]=130;
 {auto p=prefix;auto i=ids;p.sizes[0]=i.sizes[0]=std::numeric_limits<int64_t>::max()/4096+1;reject(p,i,parts,runs);}
 SpmAllocator::available=H*2-1;reject(prefix,ids,parts,runs);
 SpmAllocator::available=0;reject(prefix,ids,parts,runs);SpmAllocator::available=SpmAllocator::SPM_PLANNING_BUDGET;
 RpuKernelGraph::active=true;reject(prefix,ids,parts,runs);RpuKernelGraph::active=false;
 fail_submit=submits+2;int released=SpmAllocator::releases;bool failed=false;
 try{invoke(prefix,ids,parts,runs);}catch(const std::runtime_error&){failed=true;}
 assert(failed&&!SpmAllocator::scratch&&SpmAllocator::releases==released+1);
 fail_submit=0;auto output=invoke(prefix,ids,parts,runs);assert(output.m->bytes==BYTES&&!SpmAllocator::scratch);
}
'''
    _run_preassembled_parts_native_fixture(tmp_path, cpp, extended_checks=_extended_checks)
