"""Real adapter/capture/planner boundaries with CPU native doubles; not HW evidence."""
from types import SimpleNamespace

import pytest
import torch

import rpu_backend
from rpu_backend.adapters.qwen3_5 import text
from test_graph_cache_freeze import _cache
from test_qwen35_core_profiles import graph_factories, text_install_boundary


@pytest.fixture
def runtime(monkeypatch):
    caches = []
    policy = rpu_backend.graph.GraphRuntimePolicy()
    def graph_cache(*args, **kwargs):
        cache = _cache()
        cache._runtime_policy = kwargs.get("runtime_policy", policy)
        cache._impl.cache_invariant_ok = lambda: True
        if "max_entries" in kwargs:
            cache._impl.max_entries = lambda: kwargs["max_entries"]
        caches.append(cache)
        return cache
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", graph_cache)
    monkeypatch.setattr(rpu_backend.graph, "GraphSignature", lambda **kw: SimpleNamespace(
        _impl=(kw["op_id"], tuple(kw["shapes"]), tuple(kw["dyn_dims"])), op_id=kw["op_id"]))
    debug = [False]
    monkeypatch.setattr(torch.rpu, "get_debug_export", lambda: debug[0])
    state = SimpleNamespace(handle=7, graph_cache=graph_cache(), prefill_graph_cache=graph_cache(max_entries=1),
        public_retained_prefill=True, num_layers=24, hidden_size=4, rotary_dim=4,
        rope_theta=10000, mrope_section=(1, 1, 0), prefill_rope_cache=None, prefill_rope_shape=None)
    owner = SimpleNamespace(_rpu_qwen3_5=state)
    kv = SimpleNamespace(position=0, k_caches=[], v_caches=[], gdn_states=[], conv_states=[])
    kv.update_position = lambda length: setattr(kv, "position", kv.position + length)
    native, events = {}, []
    original_to = torch.Tensor.to

    def upload(tensor, *args, **kwargs):
        target = kwargs.get("device")
        positional = bool(args and isinstance(args[0], (str, torch.device)))
        if target is None and positional:
            target = args[0]
        if str(target).split(":")[0] == "rpu":
            # Device allocation/publication is a boundary double, not HW proof.
            assert not any(graph.state() in (1, 3) for cache in caches
                           for graph in cache._impl.entries.values())
            events.append(("upload", tuple(tensor.shape)))
            options = dict(kwargs)
            options.pop("device", None)
            return original_to(tensor, *(args[1:] if positional else args), **options).clone()
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", upload)
    def valid(handle, length):
        events.append(("valid", length)); native["length"] = length
    def rope(handle, cos, sin):
        if native.get("shape") not in (None, tuple(cos.shape)):
            assert all(cache.size() == 0 for cache in caches[1:]), "RoPE reallocated before graph retirement"
        native.update(shape=tuple(cos.shape), cos=cos.clone())
        events.append(("rope", tuple(cos.shape)))
    def forward(handle, hidden, *args):
        events.append(("forward", hidden.shape[1]))
        return hidden + native["cos"].sum(-1).reshape(1, -1, 1)
    for name, fn in (("qwen3_5_set_valid_prefill_len", valid), ("qwen3_5_set_prefill_rope", rope),
                     ("qwen3_5_forward", forward), ("qwen3_5_get_resolved_chunk_size", lambda _: 64)):
        monkeypatch.setattr(torch.ops.rpu, name, fn, raising=False)
    original = text._build_interleaved_mrope_cos_sin
    monkeypatch.setattr(text, "_build_interleaved_mrope_cos_sin",
                        lambda *a, **kw: original(*a, **kw, device="cpu"))
    plan = SimpleNamespace(graph_mode="RETAINED_CACHE", graph_key_words=lambda: (7,),
        selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(3, 7))))
    def run(length=65, *, execution=128, value=0, positions=None):
        kv.position = 0
        return text.run_qwen3_5_text(owner, torch.full((1, length, 4), value, dtype=torch.float16), kv,
            position_ids=positions, preflight_plan=(execution, 64, plan))
    return SimpleNamespace(state=state, owner=owner, kv=kv, events=events, native=native,
                           caches=caches, run=run, debug=debug)


def test_same_signature_replays_changed_input_and_position_values(runtime):
    f = runtime
    first = f.run()
    again = f.run()
    changed = f.run(value=1)
    pos = torch.arange(65).expand(3, -1).clone() + 5
    other = f.run(positions=pos)
    assert torch.equal(first, again) and torch.equal(changed, first + 1)
    assert not torch.equal(first, other)
    assert f.state.prefill_graph_cache.size() == 1 and f.kv.position == 65
    assert [replay for _, replay in f.state.prefill_graph_cache._impl.touches] == [False, True, True, True]
    assert len([event for event in f.events if event[0] == "rope"]) == 4


@pytest.mark.parametrize("input_device", ["cpu", "rpu"])
@pytest.mark.parametrize("multimodal", [False, True])
def test_public_embedding_reads_dirty_ids_before_retained_capture(
        runtime, monkeypatch, input_device, multimodal):
    from rpu_backend.adapters import qwen3_5 as dense
    from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache
    from rpu_backend.runtime import hw_attrs

    class RpuIds(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="rpu")

    f = runtime
    base = torch.ones((1, 64), dtype=torch.long)
    ids = base if input_device == "cpu" else base.as_subclass(RpuIds)
    alias, version = base.numpy(), ids._version
    table = torch.arange(6 * 4, dtype=torch.float16).reshape(6, 4) / 4
    weight = torch.arange(8 * 4, dtype=torch.float16).reshape(8, 4) / 8
    uploads, acquired_ids = [], []

    def embedding(actual_ids):
        # CPU gather/publication is a native delegate here, not an SDK proof.
        assert actual_ids.device.type == "rpu"
        assert (actual_ids is ids) == (input_device == "rpu")
        assert torch.equal(actual_ids.as_subclass(torch.Tensor), base)
        assert all(graph.state() not in (1, 3)
                   for graph in f.state.prefill_graph_cache._impl.entries.values())
        if actual_ids.device.type == "rpu":
            acquired_ids.append(actual_ids)
        hidden = torch.nn.functional.embedding(actual_ids.as_subclass(torch.Tensor), table)
        uploads.append(hidden)
        return hidden

    f.owner.embed_tokens = embedding
    f.state.execution_topology = SimpleNamespace(num_cores=8, lm_head_tp=8)
    f.state.linear_acc32 = False
    model = SimpleNamespace(model=SimpleNamespace(language_model=f.owner, rope_deltas=None),
                            _lm_head_w_rpu_keepalive=weight)
    cache = object.__new__(Qwen3_5Cache)
    cache.__dict__.update(vars(f.kv))
    cache.update_position = lambda length: setattr(cache, "position", cache.position + length)
    plan = SimpleNamespace(graph_mode="RETAINED_CACHE", graph_key_words=lambda: (7,),
        selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(3, 7))))
    monkeypatch.setattr(hw_attrs, "mark_first_forward_done", lambda _: None)
    monkeypatch.setattr(dense, "_preflight_qwen3_5_text", lambda *a, **kw: (64, 64, plan))
    monkeypatch.setattr(dense, "_prepare_qwen3_5_multimodal_spm", lambda *a: False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_mrope_position_delta", lambda *a: None, raising=False)
    fusion_inputs = []

    def fuse(actual_model, hidden, actual_ids, *_args, **_kwargs):
        assert actual_model is model and actual_ids is ids
        fusion_inputs.append(actual_ids)
        return hidden, None

    monkeypatch.setattr(dense, "fuse_visual_embeds", fuse)
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation",
                        lambda flat, *_: torch.nn.functional.linear(flat, weight), raising=False)
    original_to = torch.Tensor.to

    def upload_ids(tensor, *args, **kwargs):
        if tensor is ids and ((args and args[0] == "rpu") or kwargs.get("device") == "rpu"):
            return tensor.clone().as_subclass(RpuIds)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", upload_ids)
    outputs, snapshots = [], []
    for value in (1, 3, 1):
        alias.fill(value)
        assert ids._version == version
        cache.position = 0
        image_args = dict(pixel_values=torch.ones(4, 4), image_grid_thw=torch.tensor([[1, 2, 2]]),
                          mm_token_type_ids=torch.zeros_like(base)) if multimodal else {}
        result = dense._rpu_qwen3_5_forward.__wrapped__(
            model, input_ids=ids, past_key_values=cache, logits_to_keep=1, **image_args)
        expected_hidden = torch.nn.functional.embedding(base, table)
        expected_hidden = expected_hidden + f.native["cos"].sum(-1).reshape(1, -1, 1)
        expected = torch.nn.functional.linear(expected_hidden[:, -1:], weight).float()
        assert torch.equal(result.logits, expected) and cache.position == 64
        assert result.logits.device.type == "cpu" and result.logits.dtype == torch.float32
        outputs.append(result.logits)
        snapshots.append(result.logits.clone())
        assert all(torch.equal(old, saved) for old, saved in zip(outputs, snapshots))
    assert torch.equal(outputs[0], outputs[2]) and not torch.equal(outputs[0], outputs[1])
    assert len({hidden.data_ptr() for hidden in uploads}) == 3
    assert len(acquired_ids) == 3
    assert len(fusion_inputs) == (3 if multimodal else 0)
    assert f.state.prefill_graph_cache.size() == 1
    assert [replay for _, replay in f.state.prefill_graph_cache._impl.touches] == [False, True, True]


def test_public_embedding_hidden_keeps_native_mutable_dma_owner_contract():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src/core/fused_model_base.cpp").read_text()
    refresh = source.index("current_hidden_in_src_base() =", source.index("// Layer-0 mutable DMA"))
    retain = source.index("RpuKernelGraph::active().keep_alive(hidden_states)", refresh)
    assert "RpuGetDevAddr(hidden_states.data_ptr())" in source[refresh:retain]
    emit = source.split("void FusedModelBase::emit_layer_input_dma(", 1)[1].split(
        "// Inner layers read", 1)[0]
    assert "rpu_launch_ddr_broadcast_spm_dma_mutable(" in emit
    assert "current_hidden_in_src_base(), *pimpl_->ctx_.hidden_states" in emit


@pytest.mark.parametrize("family", ["dense", "moe"])
@pytest.mark.parametrize("length,execution", [(13, 64), (65, 128), (64, 64)])
def test_prefill_cpu_padding_keeps_current_values_and_hidden_owner_until_graph_exit(
        runtime, monkeypatch, family, length, execution):
    import gc
    import weakref
    from contextlib import contextmanager
    from rpu_backend.adapters.qwen3_5_moe import text as moe_text

    f = runtime
    module = text if family == "dense" else moe_text
    run = module.run_qwen3_5_text if family == "dense" else module.run_qwen3_5_moe_text
    f.state.mrope_section = ()
    f.state.last_a6_plan = None
    if family == "moe":
        f.owner._rpu_qwen3_5_moe = f.state
        f.state.prefill_graph = object()
    plan = SimpleNamespace(graph_mode="RETAINED_CACHE" if family == "dense" else "BOUNDED_ONESHOT",
        graph_key_words=lambda: (7,),
        selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(3, 7))))
    active, captures, cat_inputs, owner_refs = [], [], [], []

    @contextmanager
    def capture(*_args, **_kwargs):
        active.append(True)
        yield
        gc.collect()
        assert owner_refs[-1]() is not None, "native input owner died before Graph exit"
        captures.append(owner_refs[-1]().clone())
        active.pop()

    if family == "dense":
        monkeypatch.setattr(text, "_prepare_qwen35_prefill_capture", lambda *_a, **_k: capture())
    else:
        monkeypatch.setattr(moe_text, "_oneshot_scope", capture)
        monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_set_valid_prefill_len", lambda *_: None, raising=False)

    class RpuHidden(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="rpu")

        def new_zeros(self, *shape, **kwargs):
            assert not active and kwargs == {"device": "cpu"}
            return torch.zeros(*shape, dtype=self.dtype, **kwargs)

    real_cat = torch.cat

    def cat(values, dim):
        assert not active and dim == 1
        original, padding = values
        assert original.device.type == "rpu" and padding.device.type == "cpu"
        assert padding.dtype == original.dtype == torch.float16
        assert padding.shape == (1, execution - length, 4) and padding.is_contiguous()
        assert not padding.count_nonzero()
        cat_inputs.append((weakref.ref(original), weakref.ref(padding)))
        # Device dispatch is delegated here; the separate compiled actual-cat
        # case verifies mixed-input bytes and force-publication boundaries.
        return real_cat([original.as_subclass(torch.Tensor), padding], dim).as_subclass(RpuHidden)

    monkeypatch.setattr(torch, "cat", cat)

    def native(_handle, hidden, *_args):
        assert active and hidden.shape == (1, execution, 4) and hidden.is_contiguous()
        assert torch.equal(hidden[:, :length].as_subclass(torch.Tensor), base)
        assert not hidden[:, length:].count_nonzero()
        owner_refs.append(weakref.ref(hidden))
        return hidden.as_subclass(torch.Tensor).clone()

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_forward" if family == "dense" else "qwen3_5_moe_forward",
                        native, raising=False)
    base = torch.zeros((1, length, 4), dtype=torch.float16)
    hidden = base.as_subclass(RpuHidden)
    version = hidden._version
    kept, snapshots = [], []
    for value in (7., -3., 7.):
        base.numpy().fill(value)
        assert hidden._version == version
        f.kv.position = 0
        out = run(f.owner, hidden, f.kv, preflight_plan=(execution, 64, plan))
        assert f.kv.position == length and out.shape == (1, length, 4)
        kept.append(out); snapshots.append(out.clone())
        assert all(torch.equal(old, saved) for old, saved in zip(kept, snapshots))
    assert torch.equal(kept[0], kept[2]) and not torch.equal(kept[0], kept[1])
    assert len(captures) == 3
    assert len(cat_inputs) == (3 if execution > length else 0)
    assert all(padding() is None for _, padding in cat_inputs)
    if execution > length:
        assert all(owner() is None for owner in owner_refs)


def test_decode_signature_reuses_full_plan_digest_and_native_keeps_descriptor(monkeypatch):
    from contextlib import contextmanager
    from test_execution_planner import _native_candidate_v2

    graph_cache = _cache()
    signatures, consumed, oracle_calls = [], [], []
    route = [101]

    def resolve(handle):
        oracle_calls.append(handle)
        return _native_candidate_v2(
            1, (16,) * 3, descriptor_version=3,
            routes=((1, route[0], 2, 0, (), 0),))

    @contextmanager
    def capture(signature):
        signatures.append(signature)
        yield

    monkeypatch.setattr(graph_cache, "capture", capture)
    monkeypatch.setattr(rpu_backend.graph, "GraphSignature", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_resolve_decode_stage_descriptor", resolve, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_planner_cache_identity",
                        lambda handle: (handle, route[0]), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_get_resolved_chunk_size", lambda _: 16, raising=False)
    state = SimpleNamespace(handle=7, graph_cache=graph_cache, num_layers=24, hidden_size=4,
        rotary_dim=4, rope_theta=10000, mrope_section=(), execution_generation=0)
    owner = SimpleNamespace(_rpu_qwen3_5=state)
    kv = SimpleNamespace(position=0, k_caches=[], v_caches=[], gdn_states=[], conv_states=[])
    kv.update_position = lambda length: setattr(kv, "position", kv.position + length)

    def forward(handle, hidden, *args):
        descriptor = args[-1]
        assert descriptor is state.decode_plan[1].selected.stage_tuple.physical_descriptor
        consumed.append(descriptor)
        return hidden

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_forward", forward, raising=False)
    hidden = torch.zeros(1, 1, 4, dtype=torch.float16)
    for _ in range(2):
        assert text.run_qwen3_5_text(owner, hidden, kv, preflight_plan=(1, 0, None)) is hidden
    first_words = state.decode_plan[1].graph_key_words()
    assert oracle_calls == [7] and consumed[0] is consumed[1]
    assert len(first_words) == 8 < len(consumed[0])
    assert signatures[0].dyn_dims == signatures[1].dyn_dims == [24, 0, *first_words]

    route[0] = 102
    text.run_qwen3_5_text(owner, hidden, kv, preflight_plan=(1, 0, None))
    assert oracle_calls == [7, 7] and consumed[2] != consumed[0]
    assert signatures[2].dyn_dims == [24, 0, *state.decode_plan[1].graph_key_words()]
    assert signatures[2].dyn_dims != signatures[0].dyn_dims
    assert owner._rpu_last_execution_plan["physical_descriptor"] is consumed[2]
    assert kv.position == 3


def _source140_public_decode(runtime, monkeypatch, family):
    """Actual callers/helper/planner/capture; device and native endpoints are doubles."""
    from contextlib import contextmanager
    import weakref

    from rpu_backend.adapters import qwen3_5 as dense, qwen3_5_moe as moe
    from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache
    from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache
    from rpu_backend.runtime import hw_attrs
    from test_execution_planner import _native_candidate_v2

    f = runtime
    adapter = dense if family == "dense" else moe
    prefix = "qwen3_5" if family == "dense" else "qwen3_5_moe"
    cache_type = Qwen3_5Cache if family == "dense" else Qwen3_5MoeCache
    if family == "moe":
        del f.owner._rpu_qwen3_5
        f.owner._rpu_qwen3_5_moe = f.state
    f.state.num_layers = 24 if family == "dense" else 40
    f.state.adaptive_mode = None
    f.state.mrope_section = ()
    f.state.execution_generation = 0
    f.state.execution_topology = SimpleNamespace(num_cores=8, lm_head_tp=8)
    f.state.linear_acc32 = False
    embedding = torch.nn.Embedding(6, 4, dtype=torch.float16)
    table = torch.arange(24, dtype=torch.float16).reshape(6, 4) / 16
    with torch.no_grad():
        embedding.weight.copy_(table)
    f.owner.embed_tokens = embedding
    head_weight = torch.arange(24, dtype=torch.float16).reshape(6, 4) / 32
    model = SimpleNamespace(
        model=SimpleNamespace(language_model=f.owner, visual=SimpleNamespace(), rope_deltas=None),
        _lm_head_w_rpu_keepalive=head_weight)
    cache = object.__new__(cache_type)
    cache.__dict__.update(position=17, k_caches=[], v_caches=[], gdn_states=[], conv_states=[])
    cache.update_position = lambda length: setattr(cache, "position", cache.position + length)
    c = SimpleNamespace(
        family=family, adapter=adapter, model=model, inner=f.owner, state=f.state,
        cache=cache, embedding=embedding, table=table, head_weight=head_weight,
        ids=torch.tensor([[1]], dtype=torch.long), force=True,
        gathers=[], public_producers=[], native_calls=[], graph_exits=[], uploads=[])

    def active_graphs():
        return [graph for graph in f.state.graph_cache._impl.entries.values()
                if graph.state() in (1, 3)]

    class SerialEndpoint:
        active = False
        exits = 0

        @contextmanager
        def execute(self):
            assert not self.active
            self.active = True
            try:
                yield 0
            finally:
                assert not active_graphs(), "capture must end inside the public serialization scope"
                self.active = False
                self.exits += 1

    serial = SerialEndpoint()
    model._execution_session = serial
    c.serial = serial
    # Preserve exact plain Parameter/Tensor types. Only this parameter's Python
    # device report is doubled; monkeypatch restores the descriptor after each test.
    parameter_device = torch.nn.Parameter.device
    monkeypatch.setattr(torch.nn.Parameter, "device", property(
        lambda parameter: SimpleNamespace(type="rpu") if parameter is embedding.weight
        else parameter_device.__get__(parameter, type(parameter))))
    monkeypatch.setattr(hw_attrs, "mark_first_forward_done", lambda _: None)
    monkeypatch.setattr(torch.rpu, "get_ddr_flush_force", lambda: c.force)

    def preflight(_inner, _cache, length, **_kwargs):
        assert serial.active and not active_graphs() and length == 1
        return 1, 0, None

    monkeypatch.setattr(adapter, "_preflight_qwen3_5_text" if family == "dense"
                        else "_preflight_qwen3_5_moe_text", preflight)
    descriptor = _native_candidate_v2(
        1, (16,) * 3, descriptor_version=3, routes=((1, 101, 2, 0, (), 0),))
    monkeypatch.setattr(torch.ops.rpu, prefix + "_resolve_decode_stage_descriptor",
                        lambda _: descriptor, raising=False)
    monkeypatch.setattr(torch.ops.rpu, prefix + "_planner_cache_identity",
                        lambda handle: (handle, 101), raising=False)
    monkeypatch.setattr(torch.ops.rpu, prefix + "_get_resolved_chunk_size", lambda _: 16, raising=False)
    monkeypatch.setattr(torch.ops.rpu, prefix + "_set_mrope_position_delta", lambda *_: None, raising=False)
    if family == "moe":
        bridge = SimpleNamespace(set_mrope_position_delta=lambda _: None)
        monkeypatch.setattr(moe, "make_qwen3_5_moe_vision_text_bridge", lambda _: bridge)

    original_embedding = torch.nn.functional.embedding
    c.cpu_embedding = original_embedding
    producer_depth = 0

    def public_embedding(ids, weight, *args, **kwargs):
        nonlocal producer_depth
        assert serial.active and not active_graphs()
        if producer_depth == 0:
            c.public_producers.append(ids.as_subclass(torch.Tensor).detach().clone())
        producer_depth += 1
        try:
            return original_embedding(ids.as_subclass(torch.Tensor), weight, *args, **kwargs)
        finally:
            producer_depth -= 1

    monkeypatch.setattr(torch.nn.functional, "embedding", public_embedding)
    # This math-identical spy is an explicit host library endpoint. Align only
    # the captured functional delegate with it; execute the real admission helper.
    from rpu_backend.api import qwen3_5_inputs
    monkeypatch.setattr(qwen3_5_inputs, "_DECODE_FUNCTIONAL_EMBEDDING", public_embedding)

    def cpu_gather(weight, ids, padding_idx, scale_grad_by_freq, sparse):
        assert serial.active and not active_graphs()
        assert type(ids) is torch.Tensor and ids.device.type == "cpu"
        assert weight is embedding.weight and weight.dtype == torch.float16
        out = original_embedding(ids, weight.detach(), padding_idx, None, 2.,
                                 scale_grad_by_freq, sparse)
        assert type(out) is torch.Tensor and out.device.type == "cpu"
        assert out.dtype == torch.float16 and out.is_contiguous()
        c.gathers.append(out)  # Retain old fresh CPU outputs to detect overwrites.
        return out

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_decode_embedding_cpu", cpu_gather, raising=False)

    def native(_handle, hidden, *_args):
        graphs = active_graphs()
        assert serial.active and len(graphs) == 1
        graph = graphs[0]
        assert hidden.shape == (1, 1, 4) and hidden.dtype == torch.float16
        # This endpoint does computation every call; native receipt semantics
        # are tested by separate extracted-C++ tests, never reimplemented here.
        position = int(_args[5])
        current = hidden.as_subclass(torch.Tensor).detach().clone()
        c.native_calls.append((graph.state(), position, current, weakref.ref(hidden)))
        if not hasattr(graph, "_source140_end_checked"):
            original_end = graph.end

            def end():
                assert serial.active and c.native_calls[-1][3]() is not None
                original_end()
                c.graph_exits.append(graph.state())

            monkeypatch.setattr(graph, "end", end)
            graph._source140_end_checked = True
        return current + position / 32

    monkeypatch.setattr(torch.ops.rpu, prefix + "_forward", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation",
                        lambda flat, *_: torch.nn.functional.linear(
                            flat.as_subclass(torch.Tensor), head_weight), raising=False)
    original_to = torch.Tensor.to

    class RpuTokenIds(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="rpu")

    def to(tensor, *args, **kwargs):
        target = kwargs.get("device")
        positional_device = bool(args and isinstance(args[0], (str, torch.device)))
        if target is None and positional_device:
            target = args[0]
        if str(target).split(":")[0] == "rpu":
            assert serial.active and not active_graphs()
            c.uploads.append(tensor)
            remaining = args[1:] if positional_device else args
            options = dict(kwargs)
            options.pop("device", None)
            copied = original_to(tensor, *remaining, **options).clone()
            return (copied.as_subclass(RpuTokenIds)
                    if tensor.dtype == torch.int64 else copied)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    public_forward = dense._rpu_qwen3_5_forward if family == "dense" else moe._rpu_qwen3_5_moe_forward
    c.forward = lambda **kwargs: public_forward(model, past_key_values=cache,
                                               logits_to_keep=1, **kwargs)
    c.expected_logits = lambda hidden, position: torch.nn.functional.linear(
        (hidden + position / 32).reshape(-1, 4), head_weight).reshape(1, 1, 6).float()
    return c


@pytest.mark.parametrize("family", ["dense", "moe"])
@pytest.mark.parametrize("a9_stamp", [False, True])
@pytest.mark.parametrize("trace", ["all_changed", "same_then_aba"])
def test_public_decode_private_input_bridge_reads_all_current_bytes(
        runtime, monkeypatch, family, a9_stamp, trace):
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator, _VALIDATED_CLS_CACHE

    c = _source140_public_decode(runtime, monkeypatch, family)
    if a9_stamp:
        install_hw_attr_validator(c.embedding)
        assert type(c.embedding) is _VALIDATED_CLS_CACHE[torch.nn.Embedding]
    else:
        assert type(c.embedding) is torch.nn.Embedding
    ids_alias = c.ids.numpy()
    weight_alias = c.embedding.weight.detach().numpy()
    ids_version, weight_version = c.ids._version, c.embedding.weight._version
    values = ([(1, .25), (2, -.125), (1, .5), (2, -.5), (1, 0.)]
              if trace == "all_changed"
              else [(1, 0.), (1, 0.), (1, .25), (2, -.125), (1, 0.)])
    kept, snapshots, expected_hidden = [], [], []
    for token, delta in values:
        ids_alias[0, 0] = token
        # Mutate the final hidden element through a raw CPU alias: no Tensor
        # version update can stand in for reading/comparing these current bytes.
        weight_alias[token, -1] = float(c.table[token, -1]) + delta
        assert c.ids._version == ids_version and c.embedding.weight._version == weight_version
        expected = c.cpu_embedding(c.ids, c.embedding.weight.detach())
        expected_hidden.append(expected.clone())
        position = c.cache.position
        result = c.forward(input_ids=c.ids)
        assert torch.equal(c.native_calls[-1][2], expected)
        assert torch.equal(result.logits, c.expected_logits(expected, position))
        assert c.cache.position == position + 1 and result.past_key_values is c.cache
        kept.append(result.logits); snapshots.append(result.logits.clone())
        assert all(torch.equal(old, saved) for old, saved in zip(kept, snapshots))
        assert all(torch.equal(owner, old) for owner, old in zip(c.gathers, expected_hidden))
    assert len(c.gathers) == len(c.native_calls) == c.serial.exits == 5
    assert not c.public_producers and not c.uploads
    assert len({id(owner) for owner in c.gathers}) == 5
    if trace == "all_changed":
        assert all(not torch.equal(left, right)
                   for left, right in zip(expected_hidden, expected_hidden[1:]))
    else:
        assert torch.equal(expected_hidden[0], expected_hidden[1])
        assert torch.equal(expected_hidden[0], expected_hidden[-1])
        # Even equal inputs execute the original compute at the new position.
        assert not torch.equal(kept[0], kept[1])
    assert [call[0] for call in c.native_calls] == [1, 3, 3, 3, 3]
    assert c.graph_exits == [2] * 5
    assert c.state.graph_cache.size() == 1
    assert [replay for _, replay in c.state.graph_cache._impl.touches] == [False, True, True, True, True]


@pytest.mark.parametrize("family", ["dense", "moe"])
@pytest.mark.parametrize("fault", [
    "forceoff", "installed_vision", "installing_vision", "rope_delta",
    "adaptive", "action", "unbound_state", "local_hook", "local_pre_hook",
    "global_hook", "global_pre_hook", "instance_forward", "instance_call_impl",
    "dynamic_subclass", "max_norm", "cpu_ids_subclass", "rpu_ids", "inputs_embeds",
    "class_forward", "class_call", "class_call_impl", "functional_embedding",
    "class_getattribute", "class_getattr",
    "device_pre_hook", "device_custom_forward",
])
def test_public_decode_private_input_bridge_preserves_producer_fallback(
        runtime, monkeypatch, family, fault):
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator

    c = _source140_public_decode(runtime, monkeypatch, family)
    # Exercise fallback after the real A9 stamp used by installed models.
    install_hw_attr_validator(c.embedding)
    handles, offset, token_shift = [], 0., 0
    request = {"input_ids": c.ids}
    if fault == "forceoff":
        c.force = False
    elif fault in ("installed_vision", "installing_vision"):
        setattr(c.model.model.visual, "_rpu_vision_handle" if fault == "installed_vision"
                else "_rpu_vision_installing_handle", 91)
    elif fault == "rope_delta":
        c.model.model.rope_deltas = torch.tensor([[7]])
    elif fault == "adaptive":
        c.state.adaptive_mode = "adaLN"
    elif fault == "action":
        c.state.action_mode = True
    elif fault == "unbound_state":
        del c.state.adaptive_mode
    elif fault in ("local_hook", "global_hook"):
        offset = .5
        hook = lambda module, args, output: output + offset if module is c.embedding else output
        handles.append(c.embedding.register_forward_hook(hook) if fault == "local_hook"
                       else torch.nn.modules.module.register_module_forward_hook(hook))
    elif fault in ("local_pre_hook", "global_pre_hook"):
        token_shift = 1
        hook = lambda module, args: (args[0] + token_shift,) if module is c.embedding else None
        handles.append(c.embedding.register_forward_pre_hook(hook) if fault == "local_pre_hook"
                       else torch.nn.modules.module.register_module_forward_pre_hook(hook))
    elif fault == "instance_forward":
        offset = .5
        c.embedding.forward = lambda ids: torch.nn.Embedding.forward(c.embedding, ids) + offset
    elif fault == "instance_call_impl":
        offset = .5
        c.embedding._call_impl = lambda *a, **kw: torch.nn.Module._call_impl(c.embedding, *a, **kw) + offset
    elif fault == "dynamic_subclass":
        offset = .5

        class DynamicEmbedding(torch.nn.Embedding):
            def __getattribute__(self, name):
                if name == "forward":
                    return lambda ids: torch.nn.Embedding.forward(self, ids) + offset
                return super().__getattribute__(name)

        c.embedding.__class__ = DynamicEmbedding
    elif fault == "class_forward":
        offset = .5
        original_forward = torch.nn.Embedding.forward
        monkeypatch.setattr(torch.nn.Embedding, "forward",
                            lambda module, ids: original_forward(module, ids) + offset)
    elif fault == "class_call":
        offset = .5
        original_call = torch.nn.Module.__call__
        monkeypatch.setattr(torch.nn.Module, "__call__",
                            lambda module, *a, **kw: original_call(module, *a, **kw) + offset)
    elif fault == "class_call_impl":
        offset = .5
        original_impl = torch.nn.Module._call_impl
        monkeypatch.setattr(torch.nn.Module, "_call_impl",
                            lambda module, *a, **kw: original_impl(module, *a, **kw) + offset)
    elif fault == "class_getattribute":
        offset = .5
        original_forward = torch.nn.Embedding.forward
        original_lookup = torch.nn.Embedding.__getattribute__

        def current_lookup(module, name):
            if name == "forward":
                return lambda ids: original_forward(module, ids) + offset
            return original_lookup(module, name)

        monkeypatch.setattr(torch.nn.Embedding, "__getattribute__", current_lookup)
    elif fault == "class_getattr":
        original_lookup = torch.nn.Embedding.__getattr__
        monkeypatch.setattr(torch.nn.Embedding, "__getattr__",
                            lambda module, name: original_lookup(module, name))
    elif fault == "functional_embedding":
        offset = .5
        original_functional = torch.nn.functional.embedding
        monkeypatch.setattr(torch.nn.functional, "embedding",
                            lambda *a, **kw: original_functional(*a, **kw) + offset)
    elif fault == "max_norm":
        c.embedding.max_norm = .5
    elif fault == "device_pre_hook":
        def require_rpu_ids(module, args):
            assert args[0].device.type == module.weight.device.type == "rpu"

        handles.append(c.embedding.register_forward_pre_hook(require_rpu_ids))
    elif fault == "device_custom_forward":
        original_forward = c.embedding.forward

        def require_rpu_forward(ids):
            assert ids.device.type == c.embedding.weight.device.type == "rpu"
            return original_forward(ids)

        c.embedding.forward = require_rpu_forward
    elif fault in ("cpu_ids_subclass", "rpu_ids"):
        class Ids(torch.Tensor):
            @property
            def device(self):
                return SimpleNamespace(type="rpu" if fault == "rpu_ids" else "cpu")

        request["input_ids"] = c.ids.as_subclass(Ids)
    elif fault == "inputs_embeds":
        request = {"inputs_embeds": c.cpu_embedding(c.ids, c.embedding.weight.detach())}

    kept, snapshots = [], []
    try:
        for _ in range(2):
            position = c.cache.position
            result = c.forward(**request)
            if fault == "inputs_embeds":
                expected = request["inputs_embeds"]
            else:
                # max_norm updates the actual weight first, exactly as the
                # original Module producer; evaluate the oracle after the call.
                expected = c.cpu_embedding(c.ids + token_shift, c.embedding.weight.detach()) + offset
            assert torch.equal(c.native_calls[-1][2], expected)
            assert torch.equal(result.logits, c.expected_logits(expected, position))
            assert c.cache.position == position + 1
            kept.append(result.logits); snapshots.append(result.logits.clone())
            assert all(torch.equal(old, saved) for old, saved in zip(kept, snapshots))
        assert not c.gathers
        assert len(c.public_producers) == (0 if fault == "inputs_embeds" else 2)
        assert len(c.uploads) == (0 if fault == "rpu_ids" else 2)
        assert len(c.native_calls) == c.serial.exits == 2
        assert [call[0] for call in c.native_calls] == [1, 3]
        assert c.graph_exits == [2, 2] and c.state.graph_cache.size() == 1
        assert [replay for _, replay in c.state.graph_cache._impl.touches] == [False, True]
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("family,fault", [
    ("dense", "legacy"), ("dense", "image_grid"), ("dense", "token_types"),
    ("moe", "image_grid"), ("moe", "token_types"),
    ("dense", "image_decode"), ("moe", "image_decode"),
])
def test_public_decode_private_input_bridge_preserves_legacy_and_multimodal_admission(
        runtime, monkeypatch, family, fault):
    c = _source140_public_decode(runtime, monkeypatch, family)
    kwargs = {"input_ids": c.ids}
    if fault == "legacy":
        c.model._qwen3_5_legacy_text_only = True
    elif fault == "image_grid":
        kwargs["image_grid_thw"] = torch.tensor([[1, 1, 1]])
    elif fault == "token_types":
        kwargs["mm_token_type_ids"] = torch.zeros_like(c.ids)
    elif fault == "image_decode":
        kwargs.update(pixel_values=torch.zeros(1, 4), image_grid_thw=torch.tensor([[1, 1, 1]]),
                      mm_token_type_ids=torch.zeros_like(c.ids))
    if fault == "image_decode" or (family == "moe" and fault != "legacy"):
        error = NotImplementedError if fault == "image_decode" else ValueError
        position = c.cache.position
        with pytest.raises(error):
            c.forward(**kwargs)
        assert c.cache.position == position
        assert not c.gathers and not c.public_producers and not c.native_calls
        assert c.state.graph_cache.size() == 0
    else:
        for _ in range(2):
            position = c.cache.position
            expected = c.cpu_embedding(c.ids, c.embedding.weight.detach())
            result = c.forward(**kwargs)
            assert torch.equal(c.native_calls[-1][2], expected)
            assert torch.equal(result.logits, c.expected_logits(expected, position))
        assert not c.gathers and len(c.public_producers) == 2 and len(c.uploads) == 2
        assert [call[0] for call in c.native_calls] == [1, 3]
        assert c.graph_exits == [2, 2] and c.state.graph_cache.size() == 1


def test_logical_tail_is_keyed_even_when_physical_length_matches(runtime):
    f = runtime
    f.run(65)
    f.run(66)
    f.run(66)
    assert f.state.prefill_graph_cache.size() == 1 and f.kv.position == 66
    assert [hit for _, hit in f.state.prefill_graph_cache._impl.touches] == [False, False, True]


def test_shared_rope_reallocation_retires_normal_and_debug_then_rebuilds(runtime):
    f = runtime
    first = f.run()
    f.debug[0] = True
    f.run()
    assert f.state.prefill_graph_cache.size() == f.state.prefill_debug_graph_cache.size() == 1
    f.run(129, execution=192)
    assert f.state.prefill_graph_cache.size() == 0
    assert f.state.prefill_debug_graph_cache.size() == 1
    f.debug[0] = False
    restored = f.run()
    assert torch.equal(first, restored)
    assert f.state.prefill_debug_graph_cache.size() == 0
    assert [hit for _, hit in f.state.prefill_graph_cache._impl.touches] == [False, False]


@pytest.mark.parametrize("fault", ["length", "logical_tail", "missing_debug", "frozen_sibling"])
def test_ready_miss_fails_before_native_setters_or_sibling_mutation(runtime, fault):
    f = runtime
    f.run()
    f.state.prefill_graph_cache.freeze()
    if fault == "frozen_sibling":
        f.debug[0] = True
        # A pre-existing empty debug cache may plan, but it cannot replace the
        # backing used by the frozen normal graph.
        sibling = _cache()
        sibling._runtime_policy = f.state.graph_cache.runtime_policy
        f.state.prefill_debug_graph_cache = sibling
    elif fault == "missing_debug":
        f.debug[0] = True
    before = list(f.events)
    length = 66 if fault == "logical_tail" else (65 if fault == "missing_debug" else 129)
    with pytest.raises(RuntimeError, match="READY"):
        f.run(length, execution=128 if length < 129 else 192)
    assert f.events == before and f.state.prefill_graph_cache.size() == 1
    assert f.native["length"] == 65 and f.kv.position == 0


def test_ready_hit_keeps_graph_and_refreshes_native_values(runtime):
    f = runtime
    built = f.run()
    f.state.prefill_graph_cache.freeze()
    replayed = f.run(value=2)
    assert torch.equal(replayed, built + 2)
    assert f.state.prefill_graph_cache.is_frozen()
    assert f.state.prefill_graph_cache._impl.touches[-1][1]


@pytest.mark.parametrize("layers,width,adaptive,retained", [
    (24, 1024, None, True), (24, 2048, None, True), (32, 2560, None, True),
    (32, 4096, None, False), (64, 5120, None, False), (24, 2048, "adaLN", False),
])
def test_public_selector_preserves_large_and_action_profiles(layers, width, adaptive, retained):
    config = SimpleNamespace(num_hidden_layers=layers, hidden_size=width, adaptive_mode=adaptive)
    assert text._public_retained_prefill(config) is retained


def _cold_9b_owner(monkeypatch, *, precision="fp16", cores=8, vision=False):
    from test_qwen35_core_profiles import _config
    from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter

    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1" if vision else "0")
    model = torch.nn.Module()
    model.config = _config(9)
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.config = model.config.text_config
    if precision != "fp16":
        model.config.quant_config = {"method": precision}
        if precision == "w4a16_pgrp":
            model.config.quant_config["group_size"] = 32
        model._qwen3_5_quant_config = model.config.quant_config
    return Qwen3_5Adapter(model, rpu_execution={"model": {"num_cores": cores}})


@pytest.mark.parametrize("precision", ["fp16", "w8a16", "w4a16_pgrp"])
def test_9b_public_cold_session_selects_one_matching_prefill_cache(
        monkeypatch, graph_factories, text_install_boundary, precision):
    from rpu_backend.graph import GraphRuntimePolicy

    owner = _cold_9b_owner(monkeypatch, precision=precision)
    session = owner._execution_session
    assert session.graph_mode == "RETAINED_CACHE"
    policy = GraphRuntimePolicy(graph_arena_count=4) if precision == "fp16" else None
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_retained_prefill_graph", lambda *a: None, raising=False)
    inner = owner.model.model.language_model
    text_install_boundary.install_qwen3_5_text_for_rpu(
        inner, owner.model.config, execution_config=session.config,
        _graph_runtime_policy=policy, _public_prefill_graph_mode=session.graph_mode)
    state = inner._rpu_qwen3_5
    assert state.public_retained_prefill is True
    assert state.prefill_graph_cache.runtime_policy is state.graph_cache.runtime_policy
    assert state.prefill_graph_cache.runtime_policy.graph_arena_count == (4 if precision == "fp16" else 0)
    assert [name for name, _, _ in graph_factories] == ["GraphCache", "Graph", "GraphCache"]
    assert graph_factories[-1][1]["max_entries"] == 1


@pytest.mark.parametrize("cores,vision", [(4, False), (6, False), (8, True)])
def test_9b_reduced_and_vision_cold_sessions_remain_oneshot(monkeypatch, cores, vision):
    owner = _cold_9b_owner(monkeypatch, cores=cores, vision=vision)
    assert owner._execution_session.graph_mode == "BOUNDED_ONESHOT"


@pytest.mark.parametrize("fault", ["vision", "cores", "group_size", "mixed", "geometry", "unstamped"])
def test_9b_cold_scope_cannot_drift_into_a_different_lifecycle(monkeypatch, fault):
    from rpu_backend.adapters import qwen3_5 as adapter
    owner = _cold_9b_owner(monkeypatch, precision="w4a16_pgrp" if fault in ("group_size", "mixed", "unstamped") else "fp16")
    monkeypatch.setattr(adapter, "_claim_live_instance", lambda _: pytest.fail("scope drift claimed hardware"))
    if fault == "vision":
        monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    elif fault == "cores":
        before = owner._execution_session.config
        with pytest.raises(ValueError, match="lifecycle is immutable"):
            owner._execution_session.reconfigure({"model": {"num_cores": 4}})
        assert owner._execution_session.config == before
        return
    elif fault == "group_size":
        owner.model.config.quant_config["group_size"] = 64
    elif fault == "mixed":
        owner.model.config.quant_config = {"format_version": 2,
            "default": {"method": "w4a16_pgrp", "group_size": 32},
            "projection_overrides": {"linear_attn.in_proj_b": {"method": "fp16"}}}
    elif fault == "geometry":
        owner.model.config.text_config.head_dim = 128
    else:
        del owner.model._qwen3_5_quant_config
        with pytest.raises(adapter.RPUBackendError, match="generic HF loader"):
            adapter.Qwen3_5Adapter(owner.model)
        return
    with pytest.raises(ValueError, match="cold prefill lifecycle changed"):
        owner.to_rpu()
    assert not getattr(owner.model, "_rpu_swizzle_started", False)


@pytest.mark.parametrize("precision,arenas", [("fp16", 0), ("w8a16", 4), ("w4a16_pgrp", 4)])
def test_9b_retained_cold_resource_mismatch_precedes_graph_allocation(
        monkeypatch, graph_factories, text_install_boundary, precision, arenas):
    from rpu_backend.graph import GraphRuntimePolicy
    owner = _cold_9b_owner(monkeypatch, precision=precision)
    with pytest.raises(ValueError):
        text_install_boundary.install_qwen3_5_text_for_rpu(
            owner.model.model.language_model, owner.model.config,
            execution_config=owner._execution_session.config,
            _public_prefill_graph_mode=owner._execution_session.graph_mode,
            _graph_runtime_policy=GraphRuntimePolicy(graph_arena_count=arenas))
    assert graph_factories == []


@pytest.mark.parametrize("arenas", [0, 4])
def test_9b_normal_retained_scope_does_not_add_debug_queue(runtime, arenas):
    f = runtime
    f.state.hidden_size = 4096
    f.state.graph_cache._runtime_policy = rpu_backend.graph.GraphRuntimePolicy(graph_arena_count=arenas)
    f.run()
    f.run(value=1)
    assert [hit for _, hit in f.state.prefill_graph_cache._impl.touches] == [False, True]
    before = list(f.events)
    f.debug[0] = True
    with pytest.raises(ValueError, match="retained debug prefill"):
        f.run()
    assert f.events == before
    assert not hasattr(f.state, "prefill_debug_graph_cache")


@pytest.mark.parametrize("arenas", [0, 4])
@pytest.mark.parametrize("failure", ["ready_miss", "retire_before_rope"])
def test_9b_retained_failure_precedes_native_rope_or_valid_len(runtime, monkeypatch, arenas, failure):
    f = runtime
    f.state.hidden_size = 4096
    policy = rpu_backend.graph.GraphRuntimePolicy(graph_arena_count=arenas)
    f.state.graph_cache._runtime_policy = policy
    f.state.prefill_graph_cache._runtime_policy = policy
    f.run()
    if failure == "ready_miss":
        f.state.prefill_graph_cache.freeze()
    else:
        def fail_evict(_signature):
            raise RuntimeError("queue retirement failed")
        monkeypatch.setattr(f.state.prefill_graph_cache, "evict", fail_evict)
    before = list(f.events)
    with pytest.raises(RuntimeError, match="READY|queue retirement failed"):
        f.run(129, execution=192)
    assert f.events == before and f.kv.position == 0
    assert f.state.prefill_graph_cache.size() == 1


@pytest.mark.parametrize("family", ["dense", "moe"])
@pytest.mark.parametrize("length,execution", [(7, 64), (64, 64), (65, 128)])
def test_private_initial_prefill_gather_pads_on_cpu_before_one_upload(
        runtime, monkeypatch, family, length, execution):
    """Whole public caller/text path; native gather, SDK and compute are doubles."""
    from contextlib import contextmanager
    import weakref
    from rpu_backend.adapters import qwen3_5 as dense, qwen3_5_moe as moe
    from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache
    from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache
    from rpu_backend.runtime import hw_attrs

    f = runtime
    adapter = dense if family == "dense" else moe
    cache_type = Qwen3_5Cache if family == "dense" else Qwen3_5MoeCache
    raw = SimpleNamespace(current=0, begins=0, ends=0, aborts=0)
    if family == "moe":
        del f.owner._rpu_qwen3_5
        f.owner._rpu_qwen3_5_moe = f.state
        f.state.last_a6_plan = None

        def begin():
            assert raw.current == 0
            raw.current = 1
            raw.begins += 1
            f.events.append(("begin", None))

        def end():
            assert raw.current == 1
            raw.current = 0
            raw.ends += 1
            f.events.append(("end", None))

        def abort():
            raw.current = 0
            raw.aborts += 1

        # Graph/SDK execution is a disclosed boundary double. The production
        # _oneshot_scope and complete public caller/run_text bodies execute.
        f.state.prefill_graph = SimpleNamespace(
            begin=begin, end=end, abort=abort, state=lambda: raw.current)
        monkeypatch.setattr(moe, "make_qwen3_5_moe_vision_text_bridge", lambda _: SimpleNamespace(
            set_mrope_position_delta=lambda value: f.events.append(("delta", value))))
        monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_set_valid_prefill_len",
                            lambda _handle, length: (f.events.append(("valid", length)),
                                                     f.native.update(length=length)), raising=False)
    f.state.adaptive_mode = None
    f.state.action_mode = False
    f.state.mrope_section = ()
    f.state.execution_topology = SimpleNamespace(num_cores=8, lm_head_tp=8)
    f.state.linear_acc32 = False
    embedding = torch.nn.Embedding(6, 4, dtype=torch.float16)
    table = -torch.arange(24, dtype=torch.float16).reshape(6, 4) / 16
    with torch.no_grad():
        embedding.weight.copy_(table)
    f.owner.embed_tokens = embedding
    head = torch.arange(24, dtype=torch.float16).reshape(6, 4) / 32
    model = SimpleNamespace(model=SimpleNamespace(
        language_model=f.owner, visual=SimpleNamespace(), rope_deltas=None),
        _lm_head_w_rpu_keepalive=head)
    cache = object.__new__(cache_type)
    cache.__dict__.update(vars(f.kv))
    cache.update_position = lambda count: setattr(cache, "position", cache.position + count)
    parameter_device = torch.nn.Parameter.device
    monkeypatch.setattr(torch.nn.Parameter, "device", property(
        lambda parameter: SimpleNamespace(type="rpu") if parameter is embedding.weight
        else parameter_device.__get__(parameter, type(parameter))))
    monkeypatch.setattr(hw_attrs, "mark_first_forward_done", lambda _: None, raising=False)
    monkeypatch.setattr(torch.rpu, "get_ddr_flush_force", lambda: True)
    plan = SimpleNamespace(graph_mode="RETAINED_CACHE" if family == "dense" else "BOUNDED_ONESHOT",
        graph_key_words=lambda: (7,),
        selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(3, 7))))
    serial = SimpleNamespace(active=False)

    @contextmanager
    def execute():
        assert not serial.active
        serial.active = True
        try:
            yield 0
        finally:
            serial.active = False

    serial.execute = execute
    model._execution_session = serial

    def preflight(_owner, _cache, actual_length, **_kw):
        assert serial.active and actual_length == length
        return execution, 64, plan

    monkeypatch.setattr(adapter, "_preflight_qwen3_5_text" if family == "dense"
                        else "_preflight_qwen3_5_moe_text", preflight)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_mrope_position_delta", lambda *_: None, raising=False)
    gathers, gather_snapshots, pending, native_inputs = [], [], [], []
    ids = torch.ones((1, length), dtype=torch.long)

    def gather(weight, current_ids, padding_idx, scale_grad_by_freq, sparse):
        assert serial.active and current_ids is ids and weight is embedding.weight
        assert padding_idx == -1 and not scale_grad_by_freq and not sparse
        result = torch.nn.functional.embedding(current_ids, weight.detach())
        gathers.append(result)
        gather_snapshots.append(result.clone())
        return result

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_decode_embedding_cpu", gather, raising=False)
    real_cat = torch.cat
    cat_values = []

    def cat(parts, dim):
        assert serial.active and dim == 1
        assert all(part.device.type == "cpu" for part in parts)
        cat_values.append(tuple(part.clone() for part in parts))
        return real_cat(parts, dim=dim)

    monkeypatch.setattr(torch, "cat", cat)

    def native(_handle, hidden, *_args):
        graph = (next(graph for graph in f.state.prefill_graph_cache._impl.entries.values()
                      if graph.state() in (1, 3)) if family == "dense"
                 else f.state.prefill_graph)
        assert graph.state() in (1, 3)
        assert serial.active and hidden.shape == (1, execution, 4)
        assert hidden.dtype == torch.float16 and hidden.is_contiguous()
        assert _args[5] == 0
        out = torch.empty_like(hidden)
        native_inputs.append((weakref.ref(hidden), hidden.clone()))
        pending.append((weakref.ref(hidden), out))
        if not hasattr(graph, "_prefill_deferred_read_checked"):
            original_end = graph.end

            def end():
                assert serial.active
                for source, target in pending:
                    assert source() is not None
                    target.copy_(source())
                pending.clear()
                original_end()

            monkeypatch.setattr(graph, "end", end)
            graph._prefill_deferred_read_checked = True
        return out

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_forward" if family == "dense"
                        else "qwen3_5_moe_forward", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation",
                        lambda flat, *_: torch.nn.functional.linear(flat, head), raising=False)
    id_alias = ids.numpy()
    weight_alias = embedding.weight.detach().numpy()
    versions = ids._version, embedding.weight._version
    outputs, snapshots = [], []
    for token, delta in ((1, 0.), (3, .25), (1, 0.)):
        id_alias.fill(token)
        weight_alias[:] = table.numpy()
        weight_alias[token, -1] += delta
        assert (ids._version, embedding.weight._version) == versions
        expected_hidden = torch.nn.functional.embedding(ids, embedding.weight.detach())
        cache.position = 0
        with torch.no_grad():
            public = dense._rpu_qwen3_5_forward if family == "dense" else moe._rpu_qwen3_5_moe_forward
            result = public(model, input_ids=ids, past_key_values=cache)
        assert torch.equal(result.logits, torch.nn.functional.linear(expected_hidden, head).float())
        assert cache.position == length
        assert model.model.language_model._rpu_last_execution_plan["logical_len"] == length
        assert model.model.language_model._rpu_last_execution_plan["execution_len"] == execution
        assert f.native["length"] == length
        padded = native_inputs[-1][1]
        assert torch.equal(padded[:, :length], expected_hidden)
        assert not padded[:, length:].count_nonzero()
        outputs.append(result.logits); snapshots.append(result.logits.clone())
        assert all(torch.equal(old, saved) for old, saved in zip(outputs, snapshots))
        assert all(torch.equal(old, saved) for old, saved in zip(gathers, gather_snapshots))
    assert len(gathers) == len(native_inputs) == 3
    assert [event for event in f.events if event[0] == "upload"] == [("upload", (1, execution, 4))] * 3
    assert len(cat_values) == (3 if execution > length else 0)
    assert len({owner.data_ptr() for owner in gathers}) == 3
    assert len({owner.data_ptr() for owner in outputs}) == 3
    assert torch.equal(outputs[0], outputs[2]) and not torch.equal(outputs[0], outputs[1])
    assert not pending
    if family == "dense":
        assert f.state.prefill_graph_cache.size() == 1
        assert [replay for _, replay in f.state.prefill_graph_cache._impl.touches] == [False, True, True]
    else:
        assert raw.current == 0 and (raw.begins, raw.ends, raw.aborts) == (3, 3, 0)
        assert f.state.prefill_graph_cache.size() == f.state.graph_cache.size() == 0
        assert [event[0] for event in f.events if event[0] in
                {"upload", "valid", "begin", "end"}] == ["upload", "valid", "begin", "end"] * 3


@pytest.mark.parametrize("family", ["dense", "moe"])
@pytest.mark.parametrize("fault", [
    "missing_state", "continuation", "empty", "lazy_ids", "lazy_weight",
    "forceoff", "hook", "vision", "delta", "custom", "adaptive",
])
def test_private_initial_prefill_gather_rejects_unadmitted_producers(monkeypatch, family, fault):
    from rpu_backend.adapters import qwen3_5 as dense

    embedding = torch.nn.Embedding(6, 4, dtype=torch.float16)
    parameter_device = torch.nn.Parameter.device
    monkeypatch.setattr(torch.nn.Parameter, "device", property(
        lambda parameter: SimpleNamespace(type="rpu") if parameter is embedding.weight
        else parameter_device.__get__(parameter, type(parameter))))
    state = SimpleNamespace(handle=7, adaptive_mode=None, action_mode=False)
    inner = SimpleNamespace(**{"_rpu_qwen3_5" if family == "dense" else "_rpu_qwen3_5_moe": state})
    fusion = SimpleNamespace(language_model=inner, visual=SimpleNamespace(), rope_deltas=None)
    ids, position = torch.ones((1, 7), dtype=torch.long), 0
    if fault == "missing_state":
        delattr(inner, "_rpu_qwen3_5" if family == "dense" else "_rpu_qwen3_5_moe")
    elif fault == "continuation":
        position = 7
    elif fault == "empty":
        ids = torch.empty((1, 0), dtype=torch.long)
    elif fault == "lazy_ids":
        ids = torch._neg_view(-ids)
        assert ids.is_neg() and bool((ids == 1).all())
    elif fault == "lazy_weight":
        embedding.weight = torch.nn.Parameter(torch._neg_view(embedding.weight.detach()))
        assert embedding.weight.is_neg()
    elif fault == "hook":
        embedding.register_forward_hook(lambda *_: None)
    elif fault == "vision":
        fusion.visual._rpu_vision_handle = 9
    elif fault == "delta":
        fusion.rope_deltas = torch.tensor([0])
    elif fault == "custom":
        class CustomEmbedding(torch.nn.Embedding):
            pass
        embedding = CustomEmbedding(6, 4, dtype=torch.float16)
    elif fault == "adaptive":
        state.adaptive_mode = "action"
    monkeypatch.setattr(torch.rpu, "get_ddr_flush_force", lambda: fault != "forceoff")
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_decode_embedding_cpu",
                        lambda *_: pytest.fail("unadmitted prefill reached private gather"), raising=False)
    assert dense._ordinary_decode_embedding_cpu(fusion, embedding, ids, position) is None


@pytest.mark.parametrize("fault", ["ready", "capacity", "continuation", "index", "upload"])
def test_moe_private_initial_prefill_failure_keeps_mutable_setters_and_graph_untouched(
        runtime, monkeypatch, fault):
    """Actual public/preflight/planner path; retirement/device/native are boundaries."""
    from rpu_backend.adapters import qwen3_5_moe as moe
    from rpu_backend.adapters.qwen3_5 import cores
    from rpu_backend.adapters.qwen3_5_moe import cores as moe_cores
    from rpu_backend.adapters.qwen3_5_moe import text as moe_text
    from rpu_backend.api._execution import prepared_execution_plan_cache

    c = _source140_public_decode(runtime, monkeypatch, "moe")
    c.ids = torch.ones((1, 7), dtype=torch.long)
    c.cache.position = int(fault == "continuation")
    initial_position = c.cache.position
    c.state.last_a6_plan = None
    c.state.execution_topology = moe_cores.core_topology()
    c.state.max_seq_len, c.state.chunk_envelope = 128, (128, 64)
    c.state.padding_budget, c.state.padding_rows, c.state.exact_chunk_size = 64, "auto", None
    c.cache.execution_topology = cores.topology_tuple(c.state.execution_topology)
    c.cache.moe_execution_topology = moe_cores.full_topology_tuple(c.state.execution_topology)
    c.cache.max_seq_len = 6 if fault == "capacity" else 128
    c.cache.allocated_max_seq_len = 191
    # Lifecycle ownership is a hardware-installation boundary. Execute the real
    # capacity/topology/continuation checks and owner-local READY plan cache.
    monkeypatch.setattr(moe_text, "_qwen35_moe_retirement_ready", lambda _: True)
    events = []
    original_rope_deltas = None if fault in {"index", "upload"} else object()
    c.model.model.rope_deltas = original_rope_deltas
    monkeypatch.setattr(moe, "make_qwen3_5_moe_vision_text_bridge", lambda _: SimpleNamespace(
        set_mrope_position_delta=lambda value: events.append(("delta", value))))
    c.state.prefill_graph = SimpleNamespace(
        begin=lambda: pytest.fail("failed prefill started capture"),
        end=lambda: pytest.fail("failed prefill ended capture"),
        abort=lambda: pytest.fail("failed prefill reached capture abort"))
    for operation in ("set_valid_prefill_len", "set_prefill_rope", "forward"):
        monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_" + operation,
                            lambda *_: pytest.fail("failed prefill mutated native state"), raising=False)
    if fault in {"ready", "capacity", "continuation"}:
        monkeypatch.setattr(moe, "_preflight_qwen3_5_moe_text",
                            moe_text._preflight_qwen3_5_moe_text)
        monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_resolve_prefill_stage_domain",
                            lambda *_: pytest.fail("rejected preflight queried native domain"), raising=False)
        if fault == "ready":
            prepared = prepared_execution_plan_cache(c.inner, "prefill", None, "qwen3_5_moe")
            prepared.freeze()
            assert len(prepared) == 0
    else:
        plan = SimpleNamespace(graph_mode="BOUNDED_ONESHOT", selected=SimpleNamespace(
            stage_tuple=SimpleNamespace(physical_descriptor=(3, 7))))
        monkeypatch.setattr(moe, "_preflight_qwen3_5_moe_text", lambda *_args, **_kw: (64, 64, plan))
    if fault == "index":
        # The last index is checked by the actual CPU ATen embedding endpoint,
        # not a shape-only or first-token check mirrored from admission code.
        c.ids[0, -1] = c.embedding.num_embeddings
    if fault == "upload":
        previous_to = torch.Tensor.to

        def fail_upload(tensor, *args, **kwargs):
            target = kwargs.get("device", args[0] if args else None)
            if str(target).split(":")[0] == "rpu":
                assert tensor.shape == (1, 64, 4) and not tensor[:, 7:].count_nonzero()
                events.append(("upload", tuple(tensor.shape)))
                raise RuntimeError("device upload failed")
            return previous_to(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", fail_upload)
    error, message = {
        "ready": (RuntimeError, "execution plan READY miss"),
        "capacity": (ValueError, "cache capacity exceeded"),
        "continuation": (NotImplementedError, "only at cache position 0"),
        "index": (IndexError, "index out of range"),
        "upload": (RuntimeError, "device upload failed"),
    }[fault]
    with torch.no_grad(), pytest.raises(error, match=message):
        c.forward(input_ids=c.ids)
    assert c.cache.position == initial_position
    assert not c.public_producers and not c.native_calls and not c.uploads
    assert c.state.graph_cache.size() == c.state.prefill_graph_cache.size() == 0
    assert c.serial.exits == 1 and not c.serial.active
    if fault == "upload":
        assert len(c.gathers) == 1
        # The existing public pure-text delta reset precedes run_text/upload.
        # Only native text setters/capture are required to remain untouched here.
        assert events == [("delta", 0), ("upload", (1, 64, 4))]
        assert c.model.model.rope_deltas is None
    else:
        assert not c.gathers and not events
        assert c.model.model.rope_deltas is original_rope_deltas


@pytest.mark.parametrize("family,route", [
    ("dense", "forceoff"), ("dense", "hook"), ("dense", "inputs_embeds"),
    ("moe", "forceoff"), ("moe", "hook"), ("moe", "ids_offset"),
    ("moe", "weight_offset"), ("moe", "inputs_embeds"),
])
def test_private_initial_prefill_public_fallback_preserves_current_payload(
        runtime, monkeypatch, family, route):
    """Whole public fallback with real CPU embedding/cat; RPU labels are doubles."""
    import weakref
    from contextlib import contextmanager
    from rpu_backend.adapters import qwen3_5 as dense, qwen3_5_moe as moe

    c = _source140_public_decode(runtime, monkeypatch, family)
    adapter = dense if family == "dense" else moe
    prefix = "qwen3_5" if family == "dense" else "qwen3_5_moe"
    c.cache.position = 0
    c.ids = torch.ones((1, 7), dtype=torch.long)
    c.state.last_a6_plan = None
    c.state.mrope_section = ()
    c.force = route != "forceoff"
    if route == "ids_offset":
        c.ids = torch.ones((1, 8), dtype=torch.long)[:, 1:]
        assert c.ids.is_contiguous() and c.ids.storage_offset() != 0
    elif route == "weight_offset":
        backing = torch.empty(7, 4, dtype=torch.float16)
        backing[1:].copy_(c.embedding.weight.detach())
        c.embedding.weight.data = backing[1:]
        assert c.embedding.weight.is_contiguous() and c.embedding.weight.storage_offset() != 0
    hooks = []
    if route == "hook":
        c.embedding.register_forward_hook(lambda *_: hooks.append(True))
    plan = SimpleNamespace(graph_mode="RETAINED_CACHE" if family == "dense" else "BOUNDED_ONESHOT", selected=SimpleNamespace(
        stage_tuple=SimpleNamespace(physical_descriptor=(3, 7))))
    monkeypatch.setattr(adapter, "_preflight_qwen3_5_text" if family == "dense"
                        else "_preflight_qwen3_5_moe_text", lambda *_args, **_kw: (64, 64, plan))
    if family == "dense":
        monkeypatch.setattr(torch.ops.rpu, "qwen3_5_get_resolved_chunk_size",
                            lambda _: 64, raising=False)
    raw = SimpleNamespace(current=0, pending=[])
    events = []

    def end():
        for source, target in raw.pending:
            assert source() is not None
            target.copy_(source().as_subclass(torch.Tensor))
        raw.pending.clear()
        raw.current = 0
        events.append("end")

    def begin():
        assert raw.current == 0
        raw.current = 1
        events.append("begin")

    if family == "moe":
        c.state.prefill_graph = SimpleNamespace(
            begin=begin, end=end, abort=lambda: pytest.fail("valid fallback unexpectedly aborted"))
    else:
        @contextmanager
        def capture():
            begin()
            try:
                yield
            except BaseException:
                raw.current = 0
                raise
            else:
                end()

        # Dense Graph device execution is an explicit boundary double. The
        # positive test above separately exercises its actual retained cache.
        monkeypatch.setattr(dense.text, "_prepare_qwen35_prefill_capture", lambda *_a, **_kw: capture())
    monkeypatch.setattr(torch.ops.rpu, prefix + "_set_valid_prefill_len",
                        lambda _handle, length: events.append(("valid", length)), raising=False)

    class RpuHidden(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="rpu")

        def new_zeros(self, *shape, **kwargs):
            assert kwargs == {"device": "cpu"} and raw.current == 0
            return torch.zeros(*shape, dtype=self.dtype, **kwargs)

    uploads, public_inputs, cat_devices = [], [], []
    previous_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if str(target).split(":")[0] == "rpu":
            uploads.append(tensor.detach().as_subclass(torch.Tensor).clone())
            return tensor.as_subclass(RpuHidden)
        return previous_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)

    def public_embedding(ids, weight, *args, **kwargs):
        public_inputs.append(ids.clone())
        # Model the existing public embedding dispatch: real ATen CPU gather
        # then its ordinary full RPU publication before mandatory mixed cat.
        return c.cpu_embedding(ids.as_subclass(torch.Tensor), weight, *args, **kwargs).to("rpu")

    monkeypatch.setattr(torch.nn.functional, "embedding", public_embedding)
    from rpu_backend.api import qwen3_5_inputs
    monkeypatch.setattr(qwen3_5_inputs, "_DECODE_FUNCTIONAL_EMBEDDING", public_embedding)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_decode_embedding_cpu",
                        lambda *_: pytest.fail("unadmitted producer used private CPU gather"), raising=False)
    real_cat = torch.cat

    def cat(parts, dim):
        cat_devices.append(tuple(part.device.type for part in parts))
        return real_cat(parts, dim=dim)

    monkeypatch.setattr(torch, "cat", cat)
    hidden_snapshots = []

    def native(_handle, hidden, *_args):
        assert raw.current == 1 and c.serial.active
        assert hidden.device.type == "rpu" and hidden.shape == (1, 64, 4)
        assert _args[5] == 0
        hidden_snapshots.append(hidden.as_subclass(torch.Tensor).clone())
        out = torch.empty(tuple(hidden.shape), dtype=torch.float16)
        raw.pending.append((weakref.ref(hidden), out))
        return out

    monkeypatch.setattr(torch.ops.rpu, prefix + "_forward", native, raising=False)
    initial_weight = c.embedding.weight.detach().clone()
    outputs, snapshots = [], []
    for token, delta in ((1, 0.), (3, .25), (1, 0.)):
        c.ids.numpy().fill(token)
        c.embedding.weight.detach().numpy()[:] = initial_weight.numpy()
        c.embedding.weight.detach().numpy()[token, -1] += delta
        expected_hidden = c.cpu_embedding(c.ids, c.embedding.weight.detach())
        c.cache.position = 0
        with torch.no_grad():
            result = c.forward(**({"inputs_embeds": expected_hidden} if route == "inputs_embeds"
                                 else {"input_ids": c.ids}))
        # c.forward selects the final original logical row, not the padded row.
        expected = torch.nn.functional.linear(expected_hidden[:, -1:], c.head_weight).float()
        assert torch.equal(result.logits, expected) and c.cache.position == 7
        assert torch.equal(hidden_snapshots[-1][:, :7], expected_hidden)
        assert not hidden_snapshots[-1][:, 7:].count_nonzero()
        outputs.append(result.logits); snapshots.append(result.logits.clone())
        assert all(torch.equal(old, saved) for old, saved in zip(outputs, snapshots))
    assert not c.gathers and len(uploads) == (3 if route == "inputs_embeds" else 6)
    assert not raw.pending and raw.current == 0
    assert cat_devices == [("rpu", "cpu")] * 3
    assert len(public_inputs) == (0 if route == "inputs_embeds" else 3)
    assert hooks == ([True] * 3 if route == "hook" else [])
    assert events == [("valid", 7), "begin", "end"] * 3
    assert len({out.data_ptr() for out in outputs}) == 3
    assert torch.equal(outputs[0], outputs[2]) and not torch.equal(outputs[0], outputs[1])



@pytest.mark.parametrize("family", ["dense", "moe"])
@pytest.mark.parametrize("producer", ["custom", "hook"])
def test_custom_cpu_prefill_fallback_keeps_original_native_rejection(runtime, monkeypatch, family, producer):
    """A custom CPU producer must not inherit private gather's upload authority."""
    from contextlib import contextmanager
    from rpu_backend.adapters import qwen3_5 as dense, qwen3_5_moe as moe

    c = _source140_public_decode(runtime, monkeypatch, family)
    adapter = dense if family == "dense" else moe
    prefix = "qwen3_5" if family == "dense" else "qwen3_5_moe"
    c.ids = torch.ones((1, 7), dtype=torch.long)
    c.cache.position = 0
    c.state.last_a6_plan = None
    produced, events = [], []

    class CustomCpuEmbedding(torch.nn.Embedding):
        def forward(self, ids):
            current = c.cpu_embedding(ids.as_subclass(torch.Tensor), self.weight.detach())
            produced.append(current)
            return current

    if producer == "custom":
        custom = CustomCpuEmbedding(6, 4, dtype=torch.float16)
        custom.weight = c.embedding.weight
        c.inner.embed_tokens = custom
    else:
        def cpu_hook(_module, _inputs, output):
            current = output.as_subclass(torch.Tensor).clone()
            produced.append(current)
            return current

        c.embedding.register_forward_hook(cpu_hook)
    plan = SimpleNamespace(graph_mode="RETAINED_CACHE" if family == "dense" else "BOUNDED_ONESHOT", selected=SimpleNamespace(
        stage_tuple=SimpleNamespace(physical_descriptor=(3, 7))))
    monkeypatch.setattr(adapter, "_preflight_qwen3_5_text" if family == "dense"
                        else "_preflight_qwen3_5_moe_text", lambda *_args, **_kw: (64, 64, plan))
    if family == "moe":
        c.state.prefill_graph = SimpleNamespace(
            begin=lambda: events.append("begin"),
            end=lambda: pytest.fail("original invalid CPU fallback completed"),
            abort=lambda: events.append("abort"))
    else:
        @contextmanager
        def capture():
            events.append("begin")
            try:
                yield
            except BaseException:
                events.append("abort")
                raise
            else:
                pytest.fail("original invalid CPU fallback completed")

        monkeypatch.setattr(dense.text, "_prepare_qwen35_prefill_capture", lambda *_a, **_kw: capture())
    monkeypatch.setattr(torch.ops.rpu, prefix + "_set_valid_prefill_len",
                        lambda _handle, length: events.append(("valid", length)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_decode_embedding_cpu",
                        lambda *_: pytest.fail("custom producer reached private gather"), raising=False)

    def reject_prefill_cpu(_handle, hidden, *_args):
        assert hidden.device.type == "cpu" and hidden.shape == (1, 64, 4)
        assert torch.equal(hidden[:, :7], produced[-1]) and not hidden[:, 7:].count_nonzero()
        # Native device/type checking is a boundary double. Preserve the real
        # caller/padding/setter/one-shot abort order for an unadmitted producer.
        raise RuntimeError("CPU ingress requires ordinary causal text decode")

    monkeypatch.setattr(torch.ops.rpu, prefix + "_forward", reject_prefill_cpu, raising=False)
    with torch.no_grad(), pytest.raises(RuntimeError, match="ordinary causal text decode"):
        c.forward(input_ids=c.ids)
    assert len(produced) == 1 and not c.gathers and len(c.uploads) == 1 and c.cache.position == 0
    assert events == [("valid", 7), "begin", "abort"]
    assert c.serial.exits == 1 and c.state.graph_cache.size() == 0
