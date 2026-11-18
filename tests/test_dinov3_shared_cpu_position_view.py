"""Whole cold install/forward contract on real CPU ATen storage.

PrivateUse1 labels, native consumers, planner, Graph and retirement are explicit
boundary doubles. This proves no physical DDR/dispatch/flush/performance result.
"""
import ast
from contextlib import nullcontext
import gc
from pathlib import Path
import sys
import types
import weakref

import pytest
import torch
import torch.nn.functional as F


SOURCE = Path(__file__).resolve().parents[1] / "python/rpu_backend/adapters/dinov3.py"
FUNCTIONS = {
    "_dinov3_layers", "_padded_num_heads", "_clear_dinov3_handle_install",
    "install_dinov3_vision_for_rpu", "_make_dummy_dinov3_kv_caches",
    "_install_dinov3_vision_for_rpu_impl", "_validate_dinov3_input_bounds",
    "_dinov3_single_chunk_size", "build_position_idx",
    "_dinov3_shared_position_cpu_view_matches", "_rpu_dinov3_forward",
}


def _whole_functions():
    tree = ast.parse(SOURCE.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_DINOV3_HANDLE_INSTALL_ATTRS"
                for t in node.targets):
            nodes.append(node)
    assert {n.name for n in nodes if isinstance(n, ast.FunctionDef)} == FUNCTIONS
    # Retain the actual forward decorator; it is an explicit Session boundary.
    return ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))


class RpuOwner(torch.Tensor):
    """CPU ATen owner with a test-only RPU device label and copy spy."""
    trace = None
    mode = "alias"

    @staticmethod
    def __new__(cls, value):
        return value.as_subclass(cls)

    @property
    def device(self):
        return types.SimpleNamespace(type="rpu")

    def cpu(self, *args, **kwargs):
        value = self.as_subclass(torch.Tensor)
        if tuple(value.shape) == (4096, 2):
            self.trace["cold_position_acquires"] += 1
            if self.mode == "raise":
                raise RuntimeError("explicit cold device-conversion failure")
            if self.mode == "copy":
                return value.clone()
            if self.mode == "subclass":
                return self
            if self.mode == "offset":
                return value.flatten()[2:].reshape(4095, 2)
            if self.mode == "dtype":
                return value.float()
            if self.mode == "noncontiguous":
                return value.as_strided((4096, 2), (1, 4096))
        return value

    def copy_(self, value, non_blocking=False):
        if self.dtype == torch.int16:
            self.trace["original_rpu_position_copies"] += 1
        super().copy_(value, non_blocking=non_blocking)
        return self


class GraphBoundary:
    def __init__(self):
        self.entries = {}
        self.active = False

    def capture(self, sig):
        graph = self
        key = (tuple(sig.shapes), tuple(sig.dyn_dims))

        class Capture:
            def __enter__(self):
                assert not graph.active
                graph.active = True
                entry = graph.entries.setdefault(key, {"build": 0, "replay": 0})
                if entry["build"]:
                    entry["replay"] += 1
                else:
                    entry["build"] = 1

            def __exit__(self, *exc):
                graph.active = False
        return Capture()

    def clear(self):
        self.entries.clear()


class ResourceBoundary:
    """Strong Python owner/rollback spy; C++ storage deleter is not modeled."""
    def __init__(self, owner, handle, destroy, *, graphs, keepalive, label, handle_name):
        self.owner = weakref.ref(owner)
        self.handle, self.destroy = handle, destroy
        self.graphs, self.keepalive = graphs, keepalive
        self.failed, self.parent = None, None
        self.finalizer = types.SimpleNamespace(alive=True)

    def require_replaceable(self):
        assert self.failed is None

    def retire(self):
        for graph in self.graphs:
            graph.clear()
        self.destroy(self.handle)
        self.handle = None
        self.keepalive = ()

    def cleanup_failure(self, error, *owners):
        self.retire()
        return True


@pytest.fixture
def boundary(monkeypatch):
    trace = dict(cold_position_acquires=0, original_rpu_position_copies=0,
                 native_source_publications=[], native_consumers=[], destroyed=[])
    flags = types.SimpleNamespace(normal=False, force=True)
    RpuOwner.trace, RpuOwner.mode = trace, "alias"
    native = {}
    next_handle = iter(range(1, 100))

    def create():
        handle = next(next_handle)
        native[handle] = dict(position=RpuOwner(torch.zeros((4096, 2), dtype=torch.int16)))
        return handle

    def destroy(handle):
        trace["destroyed"].append(handle)
        native.pop(handle)

    def forward(handle, sequence, keys, values, tokens, descriptor):
        assert descriptor
        owner = native[handle]["position"]
        # The unchanged native full-owner force boundary must precede consumer.
        if flags.force:
            trace["native_source_publications"].append((owner.data_ptr(), owner.untyped_storage().nbytes()))
        pairs = owner.as_subclass(torch.Tensor)[:tokens - 5].clone()
        trace["native_consumers"].append((owner.data_ptr(), pairs))
        data = sequence.as_subclass(torch.Tensor).clone()
        channel = torch.arange(768, dtype=torch.float32)
        row_scale = ((channel % 7) + 1) / 128
        col_scale = ((channel % 11) - 5) / 128
        delta = pairs[:, 0:1].float() * row_scale + pairs[:, 1:2].float() * col_scale
        data[:, 5:, :] = (data[:, 5:, :].float() + delta).half()
        return RpuOwner(data)

    ops = types.SimpleNamespace(
        dinov3_vision_create=create, dinov3_vision_destroy=destroy,
        dinov3_vision_set_weights=lambda *a: None,
        dinov3_vision_set_chunk_envelope=lambda *a: None,
        dinov3_vision_set_rope=lambda *a: None,
        dinov3_vision_set_chunk_size=lambda *a: None,
        dinov3_vision_position_idx_keepalive=lambda h: native[h]["position"],
        dinov3_vision_forward=forward,
        dinov3_vision_get_resolved_chunk_size=lambda h: native[h]["chunk"],
        spm_alloc_reset_temporary=lambda: None,
    )
    class TorchBoundary:
        rpu = types.SimpleNamespace(get_ddr_flush=lambda: flags.normal,
                                   get_ddr_flush_force=lambda: flags.force)

        def __getattr__(self, name):
            return getattr(torch, name)

    torch_boundary = TorchBoundary()
    torch_boundary.ops = types.SimpleNamespace(rpu=ops)

    def plan(handle, tokens, configured, **kwargs):
        chunk = ((tokens + 15) // 16) * 16
        native[handle]["chunk"] = chunk
        stage = types.SimpleNamespace(compute_chunk=chunk, physical_descriptor=(tokens, chunk))
        return types.SimpleNamespace(
            selected=types.SimpleNamespace(stage_tuple=stage),
            graph_key_words=lambda: (chunk,), graph_mode="RETAINED_CACHE",
            selection_scope="EXPLICIT_HOST_PLANNER_DOUBLE", physical_plan_digest="host",
            plan_digest="host",
        )

    class Cache:
        def __init__(self, num_layers, **kwargs):
            self.max_seq_len = kwargs["max_seq_len"]
            self.k_caches = [torch.zeros(1)] * num_layers
            self.v_caches = [torch.zeros(1)] * num_layers

        def reset_to_position(self, position):
            assert position == 0

    backend = types.SimpleNamespace(graph=types.SimpleNamespace(
        GraphCache=GraphBoundary, GraphSignature=lambda **kwargs: types.SimpleNamespace(**kwargs)))
    scope = dict(torch=torch_boundary, nn=torch.nn, F=F, Any=object, RPUCache=Cache,
                 _NUM_CORES=8, _InstalledNativeResource=ResourceBoundary,
                 nullcontext=nullcontext,
                 _require_execution_process_safe=lambda: None,
                 _configured_vision_chunk=lambda owner: 0,
                 _plan_dinov3_vision_execution=plan, rpu_backend=backend,
                 execution_serialized=lambda fn: fn, RPUBackendError=RuntimeError,
                 UnsupportedModelError=RuntimeError,
                 build_dinov3_rope_tables=lambda **kwargs: (torch.zeros((14, 16)), torch.zeros((14, 16))))
    exec(compile(_whole_functions(), str(SOURCE), "exec"), scope)
    package = types.ModuleType("transformers"); package.__path__ = []
    outputs = types.ModuleType("transformers.modeling_outputs")
    outputs.BaseModelOutputWithPooling = lambda **kwargs: types.SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "transformers", package)
    monkeypatch.setitem(sys.modules, "transformers.modeling_outputs", outputs)
    real_to = torch.Tensor.to

    def test_to(value, *args, **kwargs):
        if kwargs.get("device") == "rpu":
            return RpuOwner(real_to(value, **{k: v for k, v in kwargs.items() if k != "device"}))
        return real_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", test_to)
    return types.SimpleNamespace(scope=scope, trace=trace, flags=flags, native=native)


def _model():
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(hidden_size=768, intermediate_size=3072,
        num_attention_heads=12, num_register_tokens=4, layer_norm_eps=1e-6,
        image_size=224, patch_size=16, rope_theta=100)
    # Native weight installation is a boundary double; actual CPU patch math is not.
    parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    linear = types.SimpleNamespace(weight=parameter, bias=parameter)
    layer = types.SimpleNamespace(attention=types.SimpleNamespace(q_proj=linear, k_proj=linear,
        v_proj=linear, o_proj=linear), mlp=types.SimpleNamespace(up_proj=linear, down_proj=linear),
        norm1=linear, norm2=linear)
    model.layer = [layer] * 12
    model._rpu_dinov3_gamma_attn = torch.ones((12, 768), dtype=torch.float16)
    model._rpu_dinov3_gamma_mlp = torch.ones((12, 768), dtype=torch.float16)
    model._rpu_dinov3_cls_reg = RpuOwner(torch.arange(5 * 768).reshape(5, 768).remainder(31).half() / 64)
    model._rpu_dinov3_patch_embed_w = torch.eye(768, dtype=torch.float16) / 16
    model._rpu_dinov3_patch_embed_b = torch.zeros(768, dtype=torch.float16)
    model.norm = types.SimpleNamespace(weight=RpuOwner(torch.ones(768, dtype=torch.float16)),
                                      bias=RpuOwner(torch.zeros(768, dtype=torch.float16)))
    return model


def _pixels(rows, cols, seed=0):
    size = 3 * rows * 16 * cols * 16
    return torch.arange(size).add(seed).remainder(17).reshape(1, 3, rows * 16, cols * 16).half() / 256


def _oracle(model, pixels):
    rows, cols = pixels.shape[-2] // 16, pixels.shape[-1] // 16
    # Independent patch reshape and independent row/column expansion, no actual
    # adapter helper or fake native-consumer output reused by this oracle.
    patches = pixels.reshape(1, 3, rows, 16, cols, 16).permute(0, 2, 4, 1, 3, 5).reshape(1, rows * cols, 768) / 16
    sequence = torch.cat((model._rpu_dinov3_cls_reg.as_subclass(torch.Tensor).unsqueeze(0), patches), dim=1)
    channel = torch.arange(768, dtype=torch.float32)
    row_positions = torch.arange(rows).repeat_interleave(cols).reshape(-1, 1)
    col_positions = torch.arange(cols).repeat(rows).reshape(-1, 1)
    delta = row_positions * (((channel % 7) + 1) / 128) + col_positions * (((channel % 11) - 5) / 128)
    sequence[:, 5:, :] = (sequence[:, 5:, :].float() + delta).half()
    hidden = F.layer_norm(sequence, (768,), torch.ones(768, dtype=torch.float16),
                          torch.zeros(768, dtype=torch.float16), model.config.layer_norm_eps)
    return hidden, hidden[:, 0, :].contiguous()


@pytest.mark.parametrize("mode", ["alias", "copy", "subclass", "offset", "dtype", "noncontiguous"])
def test_whole_install_forward_actual_alias_or_original_copy(boundary, mode):
    RpuOwner.mode = mode
    model = _model()
    boundary.scope["install_dinov3_vision_for_rpu"](model)
    resource = model._rpu_dinov3_retirement_state
    owner = model._rpu_dinov3_position_idx_keepalive
    assert resource.keepalive[1] is owner
    admitted = mode == "alias"
    assert (resource.keepalive[2] is not None) == admitted
    previous = []
    pointer = owner.data_ptr()
    cold_acquires = boundary.trace["cold_position_acquires"]
    for rows, cols, seed in [(14, 14, 0), (7, 14, 1), (14, 7, 2), (7, 14, 1)]:
        pixels = _pixels(rows, cols, seed)
        result = boundary.scope["_rpu_dinov3_forward"](model, pixels, _rpu_cpu_post_layer_norm=True)
        expected, pool = _oracle(model, pixels)
        assert torch.equal(result.last_hidden_state, expected)
        assert torch.equal(result.pooler_output, pool)
        assert bool(torch.isfinite(result.last_hidden_state).all())
        pairs = owner.as_subclass(torch.Tensor)[:rows * cols]
        assert torch.equal(pairs[:, 0], torch.arange(rows, dtype=torch.int16).repeat_interleave(cols))
        assert torch.equal(pairs[:, 1], torch.arange(cols, dtype=torch.int16).repeat(rows))
        assert owner.data_ptr() == pointer
        for output, saved in previous:
            assert torch.equal(output, saved)
        previous.append((result.last_hidden_state, result.last_hidden_state.clone()))
    assert boundary.trace["cold_position_acquires"] == cold_acquires == 1
    assert boundary.trace["original_rpu_position_copies"] == (0 if admitted else 4)
    assert boundary.trace["native_source_publications"] == [(pointer, 16384)] * 4
    assert sum(x["build"] for x in model._rpu_dinov3_graph_cache.entries.values()) == 2
    assert sum(x["replay"] for x in model._rpu_dinov3_graph_cache.entries.values()) == 2


@pytest.mark.parametrize("normal,force", [(True, True), (False, False), (True, False)])
def test_cold_controls_keep_original_route_without_cpu_acquire(boundary, normal, force):
    boundary.flags.normal, boundary.flags.force = normal, force
    model = _model(); boundary.scope["install_dinov3_vision_for_rpu"](model)
    assert model._rpu_dinov3_retirement_state.keepalive[2] is None
    result = boundary.scope["_rpu_dinov3_forward"](model, _pixels(14, 14), _rpu_cpu_post_layer_norm=True)
    assert torch.equal(result.last_hidden_state, _oracle(model, _pixels(14, 14))[0])
    assert boundary.trace["cold_position_acquires"] == 0
    assert boundary.trace["original_rpu_position_copies"] == 1
    assert len(boundary.trace["native_source_publications"]) == int(force)


def test_dynamic_flags_dirty_public_alias_and_restore(boundary):
    model = _model(); boundary.scope["install_dinov3_vision_for_rpu"](model)
    owner = model._rpu_dinov3_position_idx_keepalive
    alias = model._rpu_dinov3_retirement_state.keepalive[2]
    for normal, force in [(False, True), (True, True), (False, False), (False, True)]:
        boundary.flags.normal, boundary.flags.force = normal, force
        alias.fill_(-77)
        result = boundary.scope["_rpu_dinov3_forward"](model, _pixels(7, 14), _rpu_cpu_post_layer_norm=True)
        assert torch.equal(result.last_hidden_state, _oracle(model, _pixels(7, 14))[0])
        assert torch.equal(owner.as_subclass(torch.Tensor)[98:], torch.full((4096 - 98, 2), -77, dtype=torch.int16))
    assert boundary.trace["cold_position_acquires"] == 1
    assert boundary.trace["original_rpu_position_copies"] == 2
    assert len(boundary.trace["native_source_publications"]) == 3


@pytest.mark.parametrize("bad", ["fresh", "offset", "dtype", "noncontiguous", "lazyneg", "subclass"])
def test_alias_proof_rejects_non_exact_cpu_views(boundary, bad):
    owner = RpuOwner(torch.zeros((4096, 2), dtype=torch.int16))
    plain = owner.as_subclass(torch.Tensor)
    candidates = {
        "fresh": plain.clone(), "offset": plain.flatten()[2:].reshape(4095, 2),
        "dtype": plain.float(), "noncontiguous": plain.as_strided((4096, 2), (1, 4096)),
        "lazyneg": torch._neg_view(plain), "subclass": owner,
    }
    assert not boundary.scope["_dinov3_shared_position_cpu_view_matches"](owner, candidates[bad])


@pytest.mark.parametrize("bad", ["owner_offset", "excess_storage", "owner_noncontiguous"])
def test_alias_proof_rejects_incomplete_owner_range(boundary, bad):
    base = torch.zeros((4097, 2), dtype=torch.int16)
    owners = {
        "owner_offset": base[1:], "excess_storage": base[:4096],
        "owner_noncontiguous": base.flatten()[:8192].as_strided((4096, 2), (1, 4096)),
    }
    owner = RpuOwner(owners[bad])
    plain = owner.as_subclass(torch.Tensor)
    assert not boundary.scope["_dinov3_shared_position_cpu_view_matches"](owner, plain)


def test_cold_conversion_error_preserves_transaction_and_old_owner(boundary):
    model = _model(); boundary.scope["install_dinov3_vision_for_rpu"](model)
    old_handle = model._rpu_dinov3_handle
    old_resource = model._rpu_dinov3_retirement_state
    old_owner = model._rpu_dinov3_position_idx_keepalive
    RpuOwner.mode = "raise"
    with pytest.raises(RuntimeError, match="cold device-conversion"):
        boundary.scope["install_dinov3_vision_for_rpu"](model)
    assert model._rpu_dinov3_handle == old_handle
    assert model._rpu_dinov3_retirement_state is old_resource
    assert model._rpu_dinov3_position_idx_keepalive is old_owner
    assert boundary.trace["destroyed"] == [old_handle + 1]
    assert old_handle in boundary.native


def test_strong_resource_lifetime_reinstall_and_default_return(boundary):
    model = _model(); boundary.scope["install_dinov3_vision_for_rpu"](model)
    owner = model._rpu_dinov3_position_idx_keepalive
    weak = weakref.ref(owner)
    resource = model._rpu_dinov3_retirement_state
    model._rpu_dinov3_position_idx_keepalive = None
    del owner; gc.collect()
    assert weak() is not None and resource.keepalive[1] is weak()
    model._rpu_dinov3_position_idx_keepalive = weak()
    first_handle = model._rpu_dinov3_handle
    boundary.scope["install_dinov3_vision_for_rpu"](model)
    assert boundary.trace["destroyed"] == [first_handle]
    assert resource.keepalive == ()
    pixels = _pixels(14, 14)
    result = boundary.scope["_rpu_dinov3_forward"](model, pixels)
    assert result.last_hidden_state.device.type == "rpu"
    assert torch.equal(result.last_hidden_state.as_subclass(torch.Tensor), _oracle(model, pixels)[0])


def test_input_error_precedes_any_position_write(boundary):
    model = _model(); boundary.scope["install_dinov3_vision_for_rpu"](model)
    owner = model._rpu_dinov3_position_idx_keepalive
    before = owner.as_subclass(torch.Tensor).clone()
    with pytest.raises(RuntimeError, match="max_hw"):
        boundary.scope["_rpu_dinov3_forward"](model, _pixels(15, 14), _rpu_cpu_post_layer_norm=True)
    assert torch.equal(owner.as_subclass(torch.Tensor), before)
    assert boundary.trace["native_consumers"] == []
