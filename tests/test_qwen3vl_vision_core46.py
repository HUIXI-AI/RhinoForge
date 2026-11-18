"""Reduced Vision admission and actual installation with CPU device doubles."""
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from rpu_backend.adapters.qwen3_vl import vision
from test_qwen3_vl_merger_lifecycle import cpu_transfers  # noqa: F401


def _model(size):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    cfg = Qwen3VLVisionConfig(
        hidden_size=1152 if size == 8 else 1024,
        intermediate_size=4304 if size == 8 else 4096,
        depth=27 if size == 8 else 24, num_heads=16, patch_size=16,
        temporal_patch_size=2, spatial_merge_size=2,
        out_hidden_size={2: 2048, 4: 2560, 8: 4096}[size],
        deepstack_visual_indexes=[8, 16, 24] if size == 8 else [5, 11, 17])
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        return Qwen3VLVisionModel(cfg).eval()
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("size", (2, 4, 8))
def test_complete_fresh_vision_preflight_is_read_only(size):
    with FakeTensorMode():
        model = _model(size)
        model._rpu_execution = {"num_cores": 4, "vision": {"chunk_size": "auto"}}
        model._rpu_execution_graph_key_words = (0,)
        original = tuple((name, id(value), value._version) for name, value in model.named_parameters())
        vision.validate_qwen3_vl_vision_core_profile(model, num_cores=4)
        assert original == tuple((name, id(value), value._version) for name, value in model.named_parameters())
        assert not any(name.startswith("_rpu_") for block in model.blocks for name in vars(block))


@pytest.mark.parametrize("fault", ("last_weight", "last_bias", "last_dtype", "last_layout", "quant", "extra", "semantics", "activation"))
def test_bad_last_leaf_fails_before_conversion_or_native_create(monkeypatch, fault):
    with FakeTensorMode():
        model = _model(4)
        leaf = model.deepstack_merger_list[-1].linear_fc2
        first = model.blocks[0].attn.proj.weight
        if fault == "last_weight":
            leaf.weight = torch.nn.Parameter(leaf.weight[:-1])
        elif fault == "activation":
            model.blocks[-1].mlp.act_fn = torch.nn.SiLU()
        elif fault == "last_bias":
            leaf.bias = None
        elif fault == "last_dtype":
            leaf.weight = torch.nn.Parameter(leaf.weight.float())
        elif fault == "last_layout":
            leaf.weight = torch.nn.Parameter(leaf.weight.t().contiguous().t())
        elif fault == "quant":
            leaf.quantization_config = {}
        elif fault == "extra":
            leaf.register_parameter("weight_scale", torch.nn.Parameter(torch.ones(1)))
        else:
            model.deepstack_merger_list[-1].norm.eps = 1e-5
        def forbidden(*args, **kwargs):
            raise AssertionError("invalid tree reached mutation or native creation")
        monkeypatch.setattr(vision, "_convert_vision_block_weights_for_rpu", forbidden)
        monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_create", forbidden, raising=False)
        with pytest.raises(ValueError):
            vision.install_qwen3_vl_vision_for_rpu(model, num_cores=4)
        assert model.blocks[0].attn.proj.weight is first
        assert not hasattr(model.blocks[0], "_rpu_qwen3vl_vision_weights_converted")


@pytest.mark.parametrize("cores", (True, 0, 2, 6, 4.0, "4"))
def test_vision_core_budget_rejects_unknown_owners(cores):
    with pytest.raises(ValueError, match="4 or 8"):
        vision.install_qwen3_vl_vision_for_rpu(SimpleNamespace(), num_cores=cores)


def test_four_core_converter_keeps_existing_8b_physical_padding(monkeypatch):
    calls = []
    def swizzle(weight, cores=8):
        calls.append((cores, tuple(weight.shape)))
        return weight.clone()
    monkeypatch.setattr(vision, "tp_col_swizzle_mc_weight", swizzle)
    monkeypatch.setattr(vision, "tp_row_swizzle_mc_weight", swizzle)
    with FakeTensorMode():
        block = _model(8).blocks[0]
        vision._convert_vision_block_weights_for_rpu(
            block, 16, 1152, num_cores=4,
            physical_head_dim=80, physical_intermediate_size=4352)
        assert calls == [(4, (1280, 1152))] * 3 + [
            (4, (1152, 1280)), (4, (4352, 1152)), (4, (1152, 4352))]
        saved = block.attn.proj.weight
        with pytest.raises(RuntimeError, match="core count"):
            vision._convert_vision_block_weights_for_rpu(
                block, 16, 1152, num_cores=8,
                physical_head_dim=80, physical_intermediate_size=4352)
        assert block.attn.proj.weight is saved and len(calls) == 6


def test_four_core_cache_keeps_eight_physical_stripes(monkeypatch):
    from rpu_backend.api.cache import RPUCache
    monkeypatch.setattr(vision, "RPUCache", lambda **kw: RPUCache(**kw, device="cpu"))
    cache = vision._make_dummy_vision_kv_caches(1, 32, 16, 80, num_cores=4)
    assert cache.attn_tp == 4
    assert cache.physical_kv_cores == 8
    assert cache.nKVHeadChunk * cache.nKVHeadVx == 8 * 16 // 4


@pytest.mark.parametrize("installed, requested", ((4, 8), (8, 4), (4, 6), (4, 2)))
def test_fused_merger_lanes_must_equal_the_installed_vision_owner(installed, requested):
    with pytest.raises(ValueError, match="four- or eight-core"):
        vision.register_qwen3vl_vision_fused_merger(
            SimpleNamespace(_rpu_vision_num_cores=installed), cores=requested)


@pytest.mark.parametrize("installed, requested", ((4, 8), (8, 4), (4, 6), (4, 2)))
def test_eager_auxiliary_lanes_must_equal_the_installed_vision_owner(installed, requested):
    with pytest.raises(ValueError, match="four- or eight-core"):
        vision.enable_qwen3_vl_vision_merger_on_device(
            SimpleNamespace(_rpu_vision_num_cores=installed), cores=requested)


@pytest.mark.parametrize("size", (2, 8))
@pytest.mark.parametrize("mode", (None, "auto", "off"))
def test_actual_installer_binds_weight_cache_graph_and_native_owners(monkeypatch, size, mode):
    calls = []
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    def fake_module_to(module, *args, **kwargs):
        # FakeTensor's weakrefs cannot use Module._apply's swap_tensors path.
        # Preserve the actual requested dtype/device conversion on fresh leaves.
        for child in module.modules():
            for name, parameter in tuple(child._parameters.items()):
                if parameter is not None:
                    child._parameters[name] = torch.nn.Parameter(cpu_to(parameter, *args, **kwargs))
        return module
    monkeypatch.setattr(torch.nn.Module, "to", fake_module_to)
    def swizzle(weight, cores=8):
        calls.append(("swizzle", cores))
        return weight.clone()
    monkeypatch.setattr(vision, "tp_col_swizzle_mc_weight", swizzle)
    monkeypatch.setattr(vision, "tp_row_swizzle_mc_weight", swizzle)
    from rpu_backend.api.cache import RPUCache
    monkeypatch.setattr(vision, "RPUCache", lambda **kw: RPUCache(**kw, device="cpu"))
    class GraphCache:
        def __init__(self, **kwargs):
            self.policy = kwargs["runtime_policy"]
        def clear(self):
            calls.append(("clear",))
        def cache_invariant_ok(self):
            return True
    monkeypatch.setattr(vision.rpu_backend.graph, "GraphCache", GraphCache)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_create",
                        lambda cores=8: calls.append(("create", cores)) or 17, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_topology", lambda _: [4, 4, 4, 8], raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_destroy", lambda h: calls.append(("destroy", h)), raising=False)
    for name in ("set_weights", "set_chunk_envelope", "set_rope_route", "set_chunk_size"):
        monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_" + name, lambda *args: None, raising=False)
    with FakeTensorMode():
        model = _model(size)
        monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_position_idx_keepalive",
                            lambda _: torch.empty(4096, 2, dtype=torch.int16), raising=False)
        requested = (vision.rpu_backend.graph.GraphRuntimePolicy(
            execution_core_count=4, batch_snapshot_mode=mode) if mode is not None else None)
        handle = vision.install_qwen3_vl_vision_for_rpu(
            model, num_cores=4, max_seq_len=32, _allow_padded_8b=size == 8,
            **({"graph_runtime_policy": requested} if requested is not None else {}))
        try:
            assert handle == 17 and model._rpu_vision_num_cores == 4
            assert model._rpu_vision_graph_cache.policy.execution_core_count == 4
            assert model._rpu_vision_graph_cache.policy.batch_snapshot_mode == (mode or "auto")
            if requested is not None:
                assert model._rpu_vision_graph_cache.policy is requested
            assert model._rpu_vision_graph_cache in model._rpu_vision_retirement_state.graphs
            assert model._rpu_vision_kv_cache.attn_tp == 4
            assert all(row[1] == 4 for row in calls if row[0] in ("create", "swizzle"))
            assert model._rpu_vision_patch_embed_w.device.type == "cpu"
            assert model.merger.linear_fc1.weight.dtype == torch.float32
            assert not getattr(model, "_rpu_patch_embed_on_device", False)
            assert model._rpu_vision_physical_intermediate_size == (4352 if size == 8 else 4096)
        finally:
            model._rpu_vision_retirement_state.retire()
        assert calls[-2:] == [("clear",), ("destroy", 17)]


@pytest.mark.parametrize("policy", (True, "off", object()))
def test_installer_rejects_untyped_graph_policy_before_assets(policy):
    with pytest.raises(TypeError, match="graph_runtime_policy"):
        vision.install_qwen3_vl_vision_for_rpu(object(), graph_runtime_policy=policy)


def test_installer_rejects_graph_policy_core_mismatch_before_assets():
    requested = vision.rpu_backend.graph.GraphRuntimePolicy(execution_core_count=4)
    with pytest.raises(ValueError, match="core count"):
        vision.install_qwen3_vl_vision_for_rpu(object(), graph_runtime_policy=requested)


@pytest.mark.parametrize("cores", (4, 8))
def test_reduced_vision_releases_its_layout_before_eager_mergers(cpu_transfers, monkeypatch, cores):
    """Actual forward with CPU doubles: the eager merger row pass is sized from
    free SPM, so the Vision layout is released after the last image on four
    and eight cores."""
    from contextlib import contextmanager

    events = []

    class Merger:
        _rpu_merger_num_cores = cores

        def __call__(self, values):
            raise AssertionError("CPU merger used")

    class Visual(SimpleNamespace):
        def fast_pos_embed_interpolate(self, grid):
            return torch.zeros(int(grid.prod(-1).sum()), 2, dtype=torch.float16)

    @contextmanager
    def capture(signature):
        yield
        events.append("native")

    visual = Visual(
        _rpu_vision_handle=11, _rpu_vision_spatial_merge_size=2,
        _rpu_vision_num_layers=1, _rpu_vision_hidden_size=2, _rpu_vision_num_cores=cores,
        _rpu_vision_kv_cache=SimpleNamespace(k_caches=[None], v_caches=[None],
            max_seq_len=2048, reset_to_position=lambda position: None),
        _rpu_vision_graph_cache=SimpleNamespace(capture=capture),
        _rpu_vision_deepstack_hash=1, _rpu_vision_max_hw=40,
        _rpu_vision_host_fp32_patch=False, _rpu_vision_patch_embed_w=torch.eye(2).half(),
        _rpu_vision_patch_embed_b=None,
        _rpu_vision_position_idx_keepalive=torch.zeros(2048, 2).short(),
        _rpu_vision_deepstack_indexes=[0, 1], _rpu_vision_fused_merger=False,
        _rpu_vision_merger_on_device=True, _rpu_vision_aux_num_cores=cores,
        _rpu_vision_batch_enabled=False, _rpu_vision_batch_cap=1,
        merger=Merger(), deepstack_merger_list=[Merger(), Merger()],
    )
    def plan(handle, patches, batch, *args, **kwargs):
        return SimpleNamespace(selected=SimpleNamespace(execution_len=patches,
            stage_tuple=SimpleNamespace(compute_chunk=patches, physical_descriptor=(patches, batch),
                physical_metadata=(("manifest_state", 1), ("raw_attention_site_count", 1),
                                   ("capability_attention_ddr_site_count", 0)))),
            graph_key_words=lambda: (patches, batch), graph_mode="GRAPH_COMPOSITE_CHILD",
            selection_scope="TEST", physical_plan_digest="test", plan_digest="test")
    monkeypatch.setattr(vision, "_plan_qwen3_vl_vision_execution", plan)
    monkeypatch.setattr(vision, "_deepstack_output_cls", lambda: SimpleNamespace)
    monkeypatch.setattr(vision.rpu_backend.graph, "GraphSignature", lambda **kw: tuple(kw))
    current = [None]
    def forward(handle, values, keys, vals, patches, batch, descriptor):
        current[0] = patches
        return values.clone()
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_forward", forward, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_get_resolved_chunk_size",
                        lambda h: current[0], raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_pop_deepstack_snapshots",
                        lambda h: [torch.ones(current[0], 2).half() for _ in range(2)], raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary",
                        lambda: events.append("reset"), raising=False)
    def merger(module, values, **kwargs):
        events.append("merger")
        return values[::4] + 10
    monkeypatch.setattr(vision, "_merger_forward_on_device", merger)

    grid = torch.tensor([[1, 14, 14], [1, 16, 16]])
    pixels = torch.ones(int(grid.prod(-1).sum()), 2).half()
    result = vision._rpu_vision_forward(visual, pixels, grid)

    assert len(result.deepstack_features) == 2
    assert events == ["native", "native", "reset", "merger", "merger", "merger", "reset"]
