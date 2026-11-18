"""CPU doubles exercise eager-merger installation, output and retirement ownership."""
from types import SimpleNamespace
import ast
from pathlib import Path
import re

import pytest
import torch

from rpu_backend.adapters import qwen3_vl as qvl
from rpu_backend.adapters.qwen3_vl import vision
from rpu_backend.tests.acceptance.test_qwen3_vl_install_lifecycle import (  # noqa: F401
    install_runtime, _Model,
)


def test_actual_native_merger_schemas_parse_and_preserve_fp16_arity():
    source = (Path(__file__).resolve().parents[1] /
              "src/core/rpu_dispatch_registrations.inc").read_text()
    lib = torch.library.Library("qwen3vl_vision_schema_contract", "FRAGMENT")
    calls = []
    for name in ("qwen3vl_vision_set_merger_weights",
                 "qwen3vl_vision_set_deepstack_merger_weights",
                 "qwen3vl_vision_set_fp16_chunked_merger",
                 "qwen3vl_vision_set_fp16_compact_encoder",
                 "qwen3vl_vision_set_w8_compact_encoder"):
        start = source.index(f'm.def("{name}(')
        definition = source[start:source.index(");", start)]
        schema = "".join(ast.literal_eval(part) for part in
                         re.findall(r'"(?:[^"\\]|\\.)*"', definition))
        lib.define(schema)
        lib.impl(name, lambda *args: calls.append(args), "CompositeExplicitAutograd")
    ops = torch.ops.qwen3vl_vision_schema_contract
    tensor = torch.ones(1)
    ops.qwen3vl_vision_set_merger_weights(1, *([tensor] * 6), 2560, 1e-6)
    ops.qwen3vl_vision_set_deepstack_merger_weights(1, *([[tensor]] * 6))
    ops.qwen3vl_vision_set_deepstack_merger_weights(
        1, *([[tensor]] * 6), [tensor], [tensor])
    ops.qwen3vl_vision_set_fp16_chunked_merger(1, True)
    ops.qwen3vl_vision_set_fp16_compact_encoder(1, True)
    ops.qwen3vl_vision_set_w8_compact_encoder(1, True)
    assert len(calls) == 6
    assert all(value is None for value in calls[0][9:])
    assert all(value is None for value in calls[1][7:])
    assert calls[2][7:] == ([tensor], [tensor])
    assert calls[3] == (1, True)
    assert calls[4] == (1, True)
    assert calls[5] == (1, True)


def _merger(postshuffle=False):
    return SimpleNamespace(
        use_postshuffle_norm=postshuffle,
        norm=torch.nn.LayerNorm(8 if postshuffle else 2),
        linear_fc1=torch.nn.Linear(8, 8),
        linear_fc2=torch.nn.Linear(8, 3),
    )


def _visual():
    return SimpleNamespace(
        merger=_merger(), deepstack_merger_list=[_merger(True)],
        _rpu_vision_handle=11, _rpu_vision_has_dispatched=False,
        _rpu_vision_host_fp32_patch=False,
        _rpu_vision_patch_embed_w=torch.ones(3, 2, dtype=torch.float16),
        _rpu_vision_patch_embed_b=torch.ones(3, dtype=torch.float16),
    )


@pytest.fixture
def cpu_transfers(monkeypatch):
    original = torch.Tensor.to
    state = {"calls": 0, "fail_at": None, "error": RuntimeError}
    def to(tensor, *args, **kwargs):
        positional_rpu = bool(args) and isinstance(args[0], str) and args[0] == "rpu"
        if kwargs.get("device") == "rpu" or positional_rpu:
            state["calls"] += 1
            if state["calls"] == state["fail_at"]:
                raise state["error"]("transfer interrupted")
            if positional_rpu:
                args = ("cpu", *args[1:])
            else:
                kwargs["device"] = "cpu"
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(vision, "tp_col_swizzle_mc_weight", lambda value, cores, **kwargs: value.clone())
    return state


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("fail_at", [2, 6, 10])
def test_partial_aux_install_rolls_back_every_scope_and_can_retry(cpu_transfers, error, fail_at):
    visual = _visual()
    originals = [merger.linear_fc1.weight for merger in (visual.merger, *visual.deepstack_merger_list)]
    cpu_transfers.update(fail_at=fail_at, error=error)
    with pytest.raises(error, match="transfer interrupted"):
        vision.enable_qwen3_vl_vision_merger_on_device(visual)
    for merger, original in zip((visual.merger, *visual.deepstack_merger_list), originals):
        assert merger.linear_fc1.weight is original
        assert not any(name in vars(merger) for name in vision._VISION_MERGER_AUX_ATTRS)
    assert not any(name in vars(visual) for name in (
        "_rpu_patch_embed_w_rpu", "_rpu_patch_embed_b_rpu",
        "_rpu_patch_embed_on_device", "_rpu_vision_merger_on_device"))
    cpu_transfers["fail_at"] = None
    vision.enable_qwen3_vl_vision_merger_on_device(visual)
    assert visual._rpu_patch_embed_on_device and visual._rpu_vision_merger_on_device
    calls = cpu_transfers["calls"]
    visual._rpu_vision_has_dispatched = True
    vision.enable_qwen3_vl_vision_merger_on_device(visual)
    assert cpu_transfers["calls"] == calls


def test_failed_aux_completion_preserves_already_installed_merger(cpu_transfers):
    visual = _visual()
    vision._convert_vision_merger_for_rpu(visual.merger)
    old = {name: vars(visual.merger)[name] for name in vision._VISION_MERGER_AUX_ATTRS}
    cpu_transfers["fail_at"] = cpu_transfers["calls"] + 2
    with pytest.raises(RuntimeError, match="transfer interrupted"):
        vision.enable_qwen3_vl_vision_merger_on_device(visual)
    assert all(vars(visual.merger)[name] is value for name, value in old.items())
    assert not hasattr(visual.deepstack_merger_list[0], "_rpu_merger_on_device")


@pytest.mark.parametrize("fail_at", [1, 7, 14])
def test_w8_aux_install_rolls_back_scales_and_keeps_cpu_weights(cpu_transfers, fail_at):
    visual = _visual()
    original = visual.merger.linear_fc1.weight
    cpu_transfers["fail_at"] = fail_at
    with pytest.raises(RuntimeError, match="transfer interrupted"):
        vision.enable_qwen3_vl_vision_merger_on_device(visual, w8a16=True)
    assert visual.merger.linear_fc1.weight is original
    assert not hasattr(visual, "_rpu_vision_aux_w8a16")
    assert not hasattr(visual, "_rpu_vision_patch_embed_scale")
    for merger in (visual.merger, *visual.deepstack_merger_list):
        assert not any(name in vars(merger) for name in vision._VISION_MERGER_AUX_ATTRS)
    cpu_transfers["fail_at"] = None
    vision.enable_qwen3_vl_vision_merger_on_device(visual, w8a16=True)
    assert visual._rpu_patch_embed_w_rpu.dtype == torch.int8
    assert visual._rpu_vision_patch_embed_scale.dtype == torch.float16
    calls = cpu_transfers["calls"]
    vision.enable_qwen3_vl_vision_merger_on_device(visual, w8a16=True)
    assert cpu_transfers["calls"] == calls
    with pytest.raises(ValueError, match="precision.*cannot change"):
        vision.enable_qwen3_vl_vision_merger_on_device(visual)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("target", ["patch", "merger", "deepstack"])
def test_w8_invalid_aux_scale_rolls_back_entire_enable(cpu_transfers, value, target):
    visual = _visual()
    mergers = (visual.merger, *visual.deepstack_merger_list)
    originals = [(layer, layer.weight) for merger in mergers
                 for layer in (merger.linear_fc1, merger.linear_fc2)]
    weight = (visual._rpu_vision_patch_embed_w if target == "patch" else
              mergers[0 if target == "merger" else 1].linear_fc2.weight)
    with torch.no_grad():
        weight[0, 0] = value
    with pytest.raises(ValueError, match="finite positive scales"):
        vision.enable_qwen3_vl_vision_merger_on_device(visual, w8a16=True)
    assert all(layer.weight is original for layer, original in originals)
    assert not hasattr(visual, "_rpu_vision_aux_w8a16")
    assert not hasattr(visual, "_rpu_vision_patch_embed_scale")
    assert not hasattr(visual, "_rpu_patch_embed_on_device")
    assert all(not any(name in vars(merger) for name in vision._VISION_MERGER_AUX_ATTRS)
               for merger in mergers)
    with torch.no_grad():
        weight[0, 0] = 0
    vision.enable_qwen3_vl_vision_merger_on_device(visual, w8a16=True)
    assert visual._rpu_vision_aux_w8a16


def _small_vision_block():
    return SimpleNamespace(
        attn=SimpleNamespace(qkv=torch.nn.Linear(128, 384), proj=torch.nn.Linear(128, 128)),
        mlp=SimpleNamespace(linear_fc1=torch.nn.Linear(128, 256),
                            linear_fc2=torch.nn.Linear(256, 128)))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_w8_bad_encoder_source_rejected_before_any_block_conversion(monkeypatch, value):
    blocks = [_small_vision_block(), _small_vision_block()]
    visual = SimpleNamespace(blocks=blocks, config=SimpleNamespace(
        num_heads=8, hidden_size=128, intermediate_size=256,
        spatial_merge_size=2, deepstack_visual_indexes=[0], hidden_act="gelu_pytorch_tanh"))
    originals = [(layer, layer.weight) for block in blocks
                 for layer in (block.attn.qkv, block.attn.proj,
                               block.mlp.linear_fc1, block.mlp.linear_fc2)]
    with torch.no_grad():
        blocks[-1].mlp.linear_fc2.weight[0, 0] = value
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid cold source reached conversion or native creation")
    monkeypatch.setattr(vision, "_convert_vision_block_weights_for_rpu", forbidden)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_create", forbidden, raising=False)
    with pytest.raises(ValueError, match="source must be finite in FP16"):
        vision.install_qwen3_vl_vision_for_rpu(visual, w8a16=True)
    assert all(layer.weight is original for layer, original in originals)
    assert all(not hasattr(block, "_rpu_qwen3vl_vision_weights_converted") for block in blocks)
    assert not hasattr(visual, "_rpu_vision_handle")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_direct_w8_encoder_rejects_invalid_source_without_poison_marker(value):
    block = _small_vision_block()
    original = block.attn.proj.weight
    with torch.no_grad():
        block.mlp.linear_fc2.weight[0, 0] = value
    with pytest.raises(ValueError, match="source must be finite in FP16"):
        vision._convert_vision_block_weights_for_rpu(block, 8, 128, w8a16=True)
    assert block.attn.proj.weight is original
    assert not hasattr(block, "_rpu_qwen3vl_vision_weights_converted")
    assert not hasattr(block, "_rpu_q_w")


@pytest.mark.parametrize("postshuffle", [False, True])
@pytest.mark.parametrize("linear_acc32", [False, True])
@pytest.mark.parametrize("cores", [4,8])
def test_w8_eager_merger_matches_same_quantized_cpu_math(cpu_transfers, monkeypatch, postshuffle, linear_acc32, cores):
    merger = _merger(postshuffle)
    vision._convert_vision_merger_for_rpu(merger, cores=cores, w8a16=True)
    calls = []
    epilogues = []
    def linear(values, weight, scale, bias):
        calls.append((weight, scale))
        return torch.nn.functional.linear(values.float(), weight.float() * scale.float()[:, None],
                                          bias.float()).half()
    def selected_linear(values, weight, bias, partition, actual_cores, enabled, scale, gelu_erf=False):
        assert partition == 1 and actual_cores == cores and enabled is linear_acc32
        epilogues.append(gelu_erf)
        out = linear(values, weight, scale, bias)
        return torch.nn.functional.gelu(out) if gelu_erf else out
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", selected_linear, raising=False)
    x = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    normed = merger.norm(x.view(-1, 8) if postshuffle else x).reshape(-1, 8).half()
    expected = linear(normed, merger._rpu_fc1_quantized, merger._rpu_fc1_scale, merger._rpu_fc1_b)
    expected = linear(torch.nn.functional.gelu(expected), merger._rpu_fc2_quantized,
                      merger._rpu_fc2_scale, merger._rpu_fc2_b).float()
    calls.clear()
    actual = vision._merger_forward_on_device(merger, x, linear_acc32=linear_acc32)
    assert torch.equal(actual, expected)
    assert actual.shape == (3, 3) and actual.dtype == torch.float32
    assert calls == [(merger._rpu_fc1_w, merger._rpu_fc1_scale),
                     (merger._rpu_fc2_w, merger._rpu_fc2_scale)]
    assert epilogues == [True, False]


def test_w8_fused_merger_reuses_quantized_weights_and_scales(cpu_transfers, monkeypatch):
    visual = _visual()
    visual._rpu_vision_graph_cache = SimpleNamespace(clear=lambda: None)
    vision.enable_qwen3_vl_vision_merger_on_device(visual, w8a16=True)
    raw_row_weights = []
    def row_swizzle(weight, cores, dwidth):
        assert cores == 8 and dwidth == 1 and weight.dtype == torch.int8
        raw_row_weights.append(weight)
        return weight.clone()
    monkeypatch.setattr(vision, "tp_row_swizzle_mc_weight", row_swizzle)
    registered = []
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_set_merger_weights",
                        lambda *args: registered.append(args), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_set_deepstack_merger_weights",
                        lambda *args: registered.append(args), raising=False)
    vision.register_qwen3vl_vision_fused_merger(visual, w8a16=True)
    assert registered[0][3] is visual.merger._rpu_fc1_w
    assert registered[0][-2] is visual.merger._rpu_fc1_scale
    assert registered[0][-1] is visual.merger._rpu_fc2_scale
    assert registered[1][-2][0] is visual.deepstack_merger_list[0]._rpu_fc1_scale
    assert raw_row_weights[0] is visual.merger._rpu_fc2_quantized
    assert raw_row_weights[1] is visual.deepstack_merger_list[0]._rpu_fc2_quantized
    with pytest.raises(ValueError, match="precision.*cannot change"):
        vision.register_qwen3vl_vision_fused_merger(visual)


def test_w8_adaptive_grouping_keeps_large_exact_chunks_legal():
    grid = torch.tensor([[1, 28, 28], [1, 32, 32]])
    assert vision._vision_image_groups(grid, [784, 1024], batch_cap=1, cache_capacity=2048,
        exact_chunk=256, fused_merger=False, allow_chunked_single=True) == [(0, 1), (1, 1)]
    with pytest.raises(ValueError, match="cannot realize exact"):
        vision._vision_image_groups(grid, [784, 1024], batch_cap=1, cache_capacity=2048,
            exact_chunk=256, fused_merger=True)


@pytest.mark.parametrize("chunked_merger", [False, True])
@pytest.mark.parametrize("single_group", [False, True], ids=["mixed", "single-chunked"])
@pytest.mark.parametrize("keep_fp16", [False, True])
def test_w8_mixed_fused_eager_groups_keep_order_raw_hidden_and_fresh_outputs(
    cpu_transfers, monkeypatch, chunked_merger, single_group, keep_fp16,
):
    from contextlib import contextmanager

    class Visual(SimpleNamespace):
        def fast_pos_embed_interpolate(self, grid):
            return torch.zeros(int(grid.prod(-1).sum()), 2, dtype=torch.float16)
    signatures = []
    @contextmanager
    def capture(signature):
        signatures.append(signature)
        yield
    visual = Visual(
        _rpu_vision_handle=11, _rpu_vision_spatial_merge_size=2,
        _rpu_vision_num_layers=1, _rpu_vision_hidden_size=2,
        _rpu_vision_kv_cache=SimpleNamespace(k_caches=[None], v_caches=[None],
            max_seq_len=64, reset_to_position=lambda position: None),
        _rpu_vision_graph_cache=SimpleNamespace(capture=capture),
        _rpu_vision_deepstack_hash=1, _rpu_vision_max_hw=16,
        _rpu_vision_host_fp32_patch=False, _rpu_vision_patch_embed_w=torch.eye(2).half(),
        _rpu_vision_patch_embed_b=None, _rpu_vision_position_idx_keepalive=torch.zeros(64, 2).short(),
        _rpu_vision_deepstack_indexes=[0], _rpu_vision_fused_merger=True,
        _rpu_vision_fused_merger_w8a16=True, _rpu_vision_merger_on_device=True,
        _rpu_vision_chunked_merger_w8a16=chunked_merger,
        _rpu_vision_batch_enabled=False, _rpu_vision_batch_cap=1,
        merger=object(), deepstack_merger_list=[object()],
    )
    def plan(handle, patches, *args, **kwargs):
        # A full small-image chunk and a split larger image on the same handle.
        descriptor = (patches, 16)
        return SimpleNamespace(selected=SimpleNamespace(execution_len=patches, stage_tuple=SimpleNamespace(
            compute_chunk=16, physical_descriptor=descriptor)), graph_key_words=lambda: descriptor,
            graph_mode="GRAPH_COMPOSITE_CHILD", selection_scope="TEST", physical_plan_digest="test",
            plan_digest="test")
    monkeypatch.setattr(vision, "_plan_qwen3_vl_vision_execution", plan)
    monkeypatch.setattr(vision, "_deepstack_output_cls", lambda: SimpleNamespace)
    monkeypatch.setattr(vision.rpu_backend.graph, "GraphSignature", lambda **kwargs: SimpleNamespace(**kwargs))
    slots, current = {}, {}
    calls = {"merged": [], "snapshots": [], "eager": []}
    def forward(handle, values, k, v, patches, batch, descriptor):
        current.update(value=float(values[0, 0, 0]), patches=patches)
        raw, pooler, ds = slots.setdefault(patches, (
            torch.empty(1, patches, 2).half(), torch.empty(patches // 4, 3).half(),
            torch.empty(patches // 4, 3).half()))
        raw.fill_(current["value"])
        pooler.fill_(current["value"] + 10)
        ds.fill_(current["value"] + 20)
        return raw
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_forward", forward, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_get_resolved_chunk_size", lambda handle: 16, raising=False)
    def pop_merged(handle):
        calls["merged"].append(current["patches"])
        return slots[current["patches"]][1:]
    def pop_snapshots(handle):
        calls["snapshots"].append(current["patches"])
        return [slots[current["patches"]][0].squeeze(0)]
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_pop_merged", pop_merged, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_pop_deepstack_snapshots", pop_snapshots, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)
    def eager(merger, values, *, keep_fp16=False, linear_acc32=False):
        assert linear_acc32 is False
        offset = 10 if merger is visual.merger else 20
        calls["eager"].append((values.size(0), offset))
        return torch.full((values.size(0) // 4, 3), float(values[0, 0]) + offset,
                          dtype=torch.float16 if keep_fp16 else torch.float32)
    monkeypatch.setattr(vision, "_merger_forward_on_device", eager)
    images = ((32, 2),) if single_group else ((16, 1), (32, 2), (16, 3))
    pixels = torch.cat([torch.full((rows, 2), value) for rows, value in images]).half()
    grid = torch.tensor([[1, 4, rows // 4] for rows, _ in images])
    output = vision._rpu_vision_forward(
        visual, pixels, grid, _rpu_keep_merged_fp16=keep_fp16)
    assert torch.equal(output.last_hidden_state, pixels)
    expected = torch.cat([torch.full((rows // 4, 3), value + 10.)
                          for rows, value in images])
    assert torch.equal(output.pooler_output, expected)
    assert output.pooler_output.dtype == (torch.float16 if keep_fp16 else torch.float32)
    assert output.deepstack_features[0].dtype == output.pooler_output.dtype
    assert torch.equal(output.deepstack_features[0], expected + 10)
    expected_fused = [int(rows == 16 or chunked_merger) for rows, _ in images]
    expected_routes = tuple(
        "fused_w8a16" if rows == 16 else
        ("chunked_fused_w8a16" if chunked_merger else "eager_w8a16")
        for rows, _ in images)
    expected_calls = {
        "merged": [rows for (rows, _), fused in zip(images, expected_fused) if fused],
        "snapshots": [] if chunked_merger else [32],
        "eager": [] if chunked_merger else [(32, 10), (32, 20)],
    }
    assert [sig.dyn_dims[2] for sig in signatures] == expected_fused
    assert visual._rpu_last_execution_plan["dispatch_merger_routes"] == expected_routes
    assert calls == expected_calls
    if single_group:
        assert output.last_hidden_state.untyped_storage().data_ptr() != slots[32][0].untyped_storage().data_ptr()
    vision._rpu_vision_forward(visual, pixels + 100, grid, _rpu_keep_merged_fp16=keep_fp16)
    assert signatures[:len(images)] == signatures[len(images):]
    assert visual._rpu_last_execution_plan["dispatch_merger_routes"] == expected_routes
    assert calls == {name: values * 2 for name, values in expected_calls.items()}
    assert torch.equal(output.last_hidden_state, pixels)
    assert torch.equal(output.pooler_output, expected)
    assert torch.equal(output.deepstack_features[0], expected + 10)
    # Changing only the host snapshot dtype does not change the captured graph
    # or overwrite either previously returned result.
    switched = vision._rpu_vision_forward(
        visual, pixels, grid, _rpu_keep_merged_fp16=not keep_fp16)
    assert switched.pooler_output.dtype == (torch.float32 if keep_fp16 else torch.float16)
    assert torch.equal(switched.pooler_output, expected)
    assert signatures[:len(images)] == signatures[-len(images):]
    assert torch.equal(output.pooler_output, expected)


@pytest.mark.parametrize("snapshot_mode,error", [(1, TypeError), (None, TypeError), (True, RuntimeError)])
def test_internal_fp16_snapshot_request_rejects_invalid_state_before_native(snapshot_mode, error):
    with pytest.raises(error):
        vision._rpu_vision_forward(
            SimpleNamespace(), torch.empty(0), torch.empty(0),
            _rpu_keep_merged_fp16=snapshot_mode)


@pytest.mark.parametrize("attribute,value,error", [
    ("_rpu_vision_handle", None, RuntimeError),
    ("_rpu_vision_has_dispatched", True, RuntimeError),
    ("_rpu_vision_host_fp32_patch", True, ValueError),
])
def test_unsafe_precision_change_fails_before_any_transfer(cpu_transfers, attribute, value, error):
    visual = _visual()
    setattr(visual, attribute, value)
    with pytest.raises(error):
        vision.enable_qwen3_vl_vision_merger_on_device(visual)
    assert cpu_transfers["calls"] == 0


@pytest.mark.parametrize("postshuffle", [False, True])
@pytest.mark.parametrize("keep_fp16", [False, True])
@pytest.mark.parametrize("linear_acc32", [False, True])
def test_merger_keeps_cpu_fp32_norm_and_returns_independent_cpu_snapshot(cpu_transfers, monkeypatch, postshuffle, keep_fp16, linear_acc32):
    merger = _merger(postshuffle)
    vision._convert_vision_merger_for_rpu(merger)
    seen = []
    merger.norm.register_forward_pre_hook(lambda module, args: seen.append((args[0].dtype, args[0].shape)))
    final = torch.arange(3, dtype=torch.float16).reshape(1, 3)
    calls = []
    linear = torch.nn.functional.linear
    class DeviceResult:
        def float(self):
            pytest.fail("FP32 expansion must happen after transferring to CPU")
        def cpu(self):
            return final
    def measured_linear(values, weight, bias, partition, cores, enabled, scale, gelu_erf=False):
        assert partition == 1 and cores == 8 and enabled is linear_acc32 and scale is None
        calls.append((values.dtype, gelu_erf))
        return linear(values, weight, bias) if len(calls) == 1 else DeviceResult()
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", measured_linear, raising=False)
    # fc1 applies the GELU on the device; no host GELU runs between the GEMMs.
    monkeypatch.setattr(torch.nn.functional, "gelu",
                        lambda *a, **k: pytest.fail("merger GELU must run in the fc1 epilogue"))
    output = vision._merger_forward_on_device(
        merger, torch.arange(8, dtype=torch.float32).reshape(4, 2), keep_fp16=keep_fp16,
        linear_acc32=linear_acc32)
    assert seen == [(torch.float32, torch.Size((1, 8) if postshuffle else (4, 2)))]
    assert calls == [(torch.float16, True), (torch.float16, False)]
    assert output.dtype == (torch.float16 if keep_fp16 else torch.float32)
    assert output.device.type == "cpu"
    assert torch.equal(output.float(), final.float())
    saved = output.clone()
    final.add_(100)
    assert torch.equal(output, saved)


@pytest.mark.parametrize("fail_destroy", [False, True])
def test_composite_close_clears_child_aux_only_after_native_retirement(install_runtime, monkeypatch, fail_destroy):
    delivery_model = _Model()
    delivery_model.model.language_model._rpu_mrope_capacity = 576
    adapter = qvl.Qwen3VLAdapter(delivery_model)
    model = adapter.to_rpu()
    visual = model.model.visual
    visual.merger = _merger()
    visual.deepstack_merger_list = [_merger(True)]
    children = (visual.merger, *visual.deepstack_merger_list)
    for merger in children:
        for name in vision._VISION_MERGER_AUX_ATTRS:
            vars(merger)[name] = torch.ones(1)
    def destroy(handle):
        assert all(hasattr(merger, "_rpu_fc1_w") for merger in children)
        if fail_destroy:
            raise RuntimeError("vision destroy failed")
        install_runtime["events"].append(("destroy", "vision"))
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_destroy", destroy, raising=False)
    if fail_destroy:
        with pytest.raises(RuntimeError, match="vision destroy failed"):
            adapter.close()
        assert all(hasattr(merger, "_rpu_fc1_w") for merger in children)
    else:
        adapter.close()
        assert all(not any(name in vars(merger) for name in vision._VISION_MERGER_AUX_ATTRS)
                   for merger in children)
        assert all(merger.linear_fc1.weight.dtype == torch.float32 for merger in children)


def _model_2b():
    """Exact ordinary 2B metadata, retaining the tiny installation tensors."""
    model = _Model()
    model.config.architectures = ["Qwen3VLForConditionalGeneration"]
    model.config.tie_word_embeddings = True
    model.config.image_token_id, model.config.video_token_id = 151655, 151656
    model.config.vision_start_token_id, model.config.vision_end_token_id = 151652, 151653
    model.config.text_config = SimpleNamespace(
        model_type="qwen3_vl_text", hidden_size=2048,
        intermediate_size=6144, num_hidden_layers=28,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128,
        hidden_act="silu", rms_norm_eps=1e-6, vocab_size=151936,
        attention_bias=False, use_cache=True, tie_word_embeddings=True,
        rope_scaling={"rope_type": "default", "rope_theta": 5_000_000,
                      "mrope_interleaved": True, "mrope_section": [24, 20, 20]},
    )
    model.config.vision_config = SimpleNamespace(
        model_type="qwen3_vl", hidden_size=1024, intermediate_size=4096,
        depth=24, num_heads=16, patch_size=16, temporal_patch_size=2,
        spatial_merge_size=2, deepstack_visual_indexes=[5, 11, 17],
        hidden_act="gelu_pytorch_tanh", out_hidden_size=2048,
        in_channels=3, num_position_embeddings=2304,
    )
    return model


def _model_4b():
    model = _model_2b()
    text = model.config.text_config
    text.hidden_size, text.intermediate_size, text.num_hidden_layers = 2560, 9728, 36
    text.num_attention_heads = 32
    model.config.vision_config.out_hidden_size = 2560
    return model


def _model_8b():
    """Exact public config with the small installation doubles above."""
    model = _Model()
    model.config.tie_word_embeddings = False
    model.config.text_config = SimpleNamespace(
        model_type="qwen3_vl_text", hidden_size=4096,
        intermediate_size=12288, num_hidden_layers=36,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        hidden_act="silu", rms_norm_eps=1e-6, vocab_size=151936,
        attention_bias=False, use_cache=True,
        rope_scaling={"rope_type": "default", "rope_theta": 5_000_000,
                      "mrope_interleaved": True, "mrope_section": [24, 20, 20]},
    )
    model.config.vision_config = SimpleNamespace(
        model_type="qwen3_vl", hidden_size=1152, intermediate_size=4304,
        depth=27, num_heads=16, patch_size=16, temporal_patch_size=2,
        spatial_merge_size=2, deepstack_visual_indexes=[8, 16, 24],
        hidden_act="gelu_pytorch_tanh", out_hidden_size=4096,
        in_channels=3, num_position_embeddings=2304,
    )
    return model


@pytest.mark.parametrize(
    "factory,legacy_2b",
    [(_model_2b, True), (_model_4b, False)],
)
def test_dense_eager_profile_accepts_only_the_exact_ordinary_configs(
    factory, legacy_2b,
):
    model = factory()
    assert qvl._is_qwen3_vl_dense_eager_profile(model)
    assert qvl._is_qwen3_vl_2b_eager_profile(model) is legacy_2b


@pytest.mark.parametrize("factory", [_model_2b, _model_4b])
@pytest.mark.parametrize("scope", ["root", "text", "vision"])
@pytest.mark.parametrize("attribute", ["quant_config", "quantization_config"])
def test_dense_eager_profile_rejects_quant_metadata_at_every_config_scope(
    factory, scope, attribute,
):
    model = factory()
    owner = {
        "root": model.config,
        "text": model.config.text_config,
        "vision": model.config.vision_config,
    }[scope]
    setattr(owner, attribute, {})
    assert not qvl._is_qwen3_vl_dense_eager_profile(model)


@pytest.mark.parametrize("factory", [_model_2b, _model_4b])
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda model: setattr(model.config, "architectures", ["Other"]),
            id="architecture",
        ),
        pytest.param(
            lambda model: setattr(model.config, "tie_word_embeddings", False),
            id="root-tie",
        ),
        pytest.param(
            lambda model: setattr(model.config, "image_token_id", 151654),
            id="image-token",
        ),
        pytest.param(
            lambda model: setattr(model.config.text_config, "rms_norm_eps", 1e-5),
            id="text-rms-eps",
        ),
        pytest.param(
            lambda model: model.config.text_config.rope_scaling.update(
                rope_type="linear"
            ),
            id="text-rope",
        ),
        pytest.param(
            lambda model: setattr(
                model.config.text_config, "tie_word_embeddings", False
            ),
            id="text-tie",
        ),
        pytest.param(
            lambda model: setattr(model.config.vision_config, "model_type", "other"),
            id="vision-model-type",
        ),
        pytest.param(
            lambda model: setattr(
                model.config.vision_config, "num_position_embeddings", 2303
            ),
            id="vision-position-embeddings",
        ),
        pytest.param(
            lambda model: setattr(
                model.config.vision_config,
                "deepstack_visual_indexes",
                [5, 11, 16],
            ),
            id="vision-deepstack",
        ),
    ],
)
def test_dense_eager_profile_rejects_semantic_drift(factory, mutate):
    model = factory()
    mutate(model)
    assert not qvl._is_qwen3_vl_dense_eager_profile(model)


@pytest.fixture
def eager_install_runtime(install_runtime, cpu_transfers, monkeypatch):
    """Run the real auxiliary converter from the real top-level install."""
    from rpu_backend.quant import load as quant_load
    # This fixture deliberately has no decoder layers. Actual bank allocation,
    # swizzling and failure ownership are covered by the 2B weight-bank tests.
    monkeypatch.setattr(quant_load, "_move_materialized_decoder_for_rpu",
                        lambda model, **kwargs: None)
    installer = qvl.install_qwen3_vl_vision_for_rpu
    state = install_runtime
    state["eager_calls"] = []
    state["fused_calls"] = []
    state["vision_replay_calls"] = []

    def install_visual(model, **kwargs):
        handle = installer(model, **kwargs)
        state["vision_model"] = model
        # The native tower installer is mocked, but auxiliary tensors/conversion
        # and retirement run through production code with tiny CPU transfers.
        for name, value in vars(_visual()).items():
            if name != "_rpu_vision_handle":
                vars(model)[name] = value
        model._rpu_vision_host_fp32_patch = qvl.rpu_env_bool("RPU_QWEN3VL_VISION_HOST_FP32_PATCH")
        model._rpu_vision_num_cores = kwargs.get("num_cores", 8)
        return handle

    def enable(visual, **kwargs):
        state["eager_calls"].append(visual)
        state.setdefault("eager_kwargs", []).append(kwargs)
        assert ("install", "vision") in state["events"]
        assert ("install", "text") in state["events"]
        assert ("set_vision_linear_acc32", 11, visual._rpu_vision_linear_acc32) in state["events"]
        owner = visual._qwen3_vl_retirement_owner
        assert not owner._rpu_is_ready
        assert not visual._rpu_vision_has_dispatched
        return vision.enable_qwen3_vl_vision_merger_on_device(visual, **kwargs)

    monkeypatch.setattr(qvl, "install_qwen3_vl_vision_for_rpu", install_visual)
    monkeypatch.setattr(qvl, "enable_qwen3_vl_vision_merger_on_device", enable)
    def register_fused(visual):
        assert visual._rpu_vision_merger_on_device
        assert state["vision_kwargs"]["_pack_fp16_block_weights"] is True
        assert not state["vision_kwargs"]["_allow_padded_8b"]
        assert not visual._qwen3_vl_retirement_owner._rpu_is_ready
        assert not visual._rpu_vision_has_dispatched
        state["fused_calls"].append(visual)
        return vision.register_qwen3vl_vision_fused_merger(visual)
    monkeypatch.setattr(qvl, "register_qwen3vl_vision_fused_merger", register_fused)
    monkeypatch.setattr(vision, "tp_row_swizzle_mc_weight", lambda value, cores, **kwargs: value.clone())
    for op in ("qwen3vl_vision_set_merger_weights", "qwen3vl_vision_set_deepstack_merger_weights"):
        monkeypatch.setattr(torch.ops.rpu, op,
                            lambda handle, *args, op=op: state["events"].append((op, handle)),
                            raising=False)
    for name in ("qwen3vl_vision_set_fast_replay", "qwen3vl_vision_set_preload_replay_skip"):
        def set_replay(handle, enabled, name=name):
            visual = state["vision_model"]
            assert handle == visual._rpu_vision_handle == 11 and enabled is True
            assert state["vision_kwargs"]["_pack_fp16_block_weights"] is True
            assert not state["vision_kwargs"]["_allow_padded_8b"]
            assert state["vision_kwargs"].get("num_cores", 8) == 8
            assert not visual._qwen3_vl_retirement_owner._rpu_is_ready
            assert not visual._rpu_vision_has_dispatched
            assert not visual._rpu_vision_graph_cache.is_frozen()
            assert ("install", "text") in state["events"]
            selected = ("set_vision_linear_acc32", 11, visual._rpu_vision_linear_acc32)
            assert selected in state["events"]
            if visual._rpu_vision_linear_acc32:
                assert state["events"].index(selected) < state["events"].index(
                    ("set_vision_large_image_auto_chunk", 11, True))
            else:
                assert ("set_vision_large_image_auto_chunk", 11, True) not in state["events"]
            assert state["vision_replay_calls"] == ([] if name.endswith("set_fast_replay") else
                                                   ["qwen3vl_vision_set_fast_replay"])
            state["vision_replay_calls"].append(name)
        monkeypatch.setattr(torch.ops.rpu, name, set_replay, raising=False)
    for name in (
        "qwen3vl_vision_set_bake_merger", "causal_decoder_set_fast_replay",
        "causal_decoder_set_preload_replay_skip",
    ):
        monkeypatch.setattr(torch.ops.rpu, name,
                            lambda *args: pytest.fail("ordinary VL8 must not enable delivery replay routes"),
                            raising=False)
    return state


@pytest.mark.parametrize("factory", [_model_2b, _model_4b, _model_8b])
@pytest.mark.parametrize("vision_acc32", [None, False, True])
def test_public_eager_aux_keeps_cold_accumulation_and_retires_it(eager_install_runtime, cpu_transfers, factory, vision_acc32):
    configured = factory()
    if vision_acc32 is not None:
        configured._rpu_execution = {"vision": {"linear_acc32": vision_acc32}}
    adapter = qvl.Qwen3VLAdapter(configured)
    model = adapter.to_rpu()
    visual = model.model.visual
    assert visual._rpu_vision_linear_acc32 is (vision_acc32 is True)
    assert eager_install_runtime["acc32"] is False
    compact = factory is not _model_8b and vision_acc32 is True
    assert eager_install_runtime["eager_calls"] == [visual]
    assert visual._rpu_vision_merger_on_device and visual._rpu_patch_embed_on_device
    assert all(merger._rpu_merger_on_device
               for merger in (visual.merger, *visual.deepstack_merger_list))
    assert bool(getattr(visual, "_rpu_vision_fused_merger", False)) is compact
    assert bool(getattr(visual, "_rpu_vision_chunked_merger_fp16", False)) is compact
    assert bool(getattr(visual, "_rpu_vision_compact_encoder_fp16", False)) is compact
    assert not getattr(visual, "_rpu_vision_compact_encoder_w8a16", False)
    assert eager_install_runtime["fused_calls"] == ([visual] if compact else [])
    cold_calls = [event for event in eager_install_runtime["events"] if event[0] == "set_fp16_chunked_merger"]
    assert cold_calls == ([("set_fp16_chunked_merger", 11, True)] if compact else [])
    if cold_calls:
        events = eager_install_runtime["events"]
        assert events.index(("qwen3vl_vision_set_deepstack_merger_weights", 11)) < events.index(cold_calls[0])
        assert events.index(cold_calls[0]) < events.index(("set_fp16_compact_encoder", 11, True))
    cap_events = [event for event in eager_install_runtime["events"]
                  if event[0] == "set_vision_large_image_auto_chunk"]
    assert cap_events == ([("set_vision_large_image_auto_chunk", 11, True)] if compact else [])
    replay_calls = ([] if factory is _model_8b else
                    ["qwen3vl_vision_set_fast_replay", "qwen3vl_vision_set_preload_replay_skip"])
    assert eager_install_runtime["vision_replay_calls"] == replay_calls
    calls = cpu_transfers["calls"]
    assert calls > 0
    assert adapter.to_rpu() is model
    assert cpu_transfers["calls"] == calls
    assert eager_install_runtime["eager_calls"] == [visual]
    assert eager_install_runtime["fused_calls"] == ([visual] if compact else [])
    assert eager_install_runtime["vision_replay_calls"] == replay_calls
    adapter.close()
    assert ("destroy", "vision") in eager_install_runtime["events"]
    assert ("destroy", "text") in eager_install_runtime["events"]
    assert not hasattr(visual, "_rpu_patch_embed_w_rpu")
    assert not hasattr(visual, "_rpu_vision_chunked_merger_fp16")
    assert not hasattr(visual, "_rpu_vision_compact_encoder_fp16")
    assert all(not any(name in vars(merger) for name in vision._VISION_MERGER_AUX_ATTRS)
               for merger in (visual.merger, *visual.deepstack_merger_list))


@pytest.mark.parametrize("host_patch", [False, True])
def test_exact_2b_text_w8_reuses_fp16_vision_aux_and_preserves_host_override(
    eager_install_runtime, cpu_transfers, monkeypatch, host_patch,
):
    import copy
    from rpu_backend.quant.load import _QWEN3_VL_2B_W8A16_QUANT_CONFIG

    model = _model_2b()
    model.config.quant_config = copy.deepcopy(_QWEN3_VL_2B_W8A16_QUANT_CONFIG)
    # This fixture omits decoder layers. Existing W8 loader tests validate the
    # real inventory; isolate selection and the actual auxiliary conversion.
    monkeypatch.setattr(qvl, "_validate_qwen3_vl_2b_w8a16_model", lambda model: True)
    monkeypatch.setattr(qvl, "_qwen3_vl_text_projection_weights_are_fp16", lambda model: False)
    monkeypatch.setenv("RPU_QWEN3VL_VISION_HOST_FP32_PATCH", str(int(host_patch)))
    owner = qvl.Qwen3VLAdapter(model)
    assert owner.to_rpu() is model
    visual = model.model.visual
    assert eager_install_runtime["eager_calls"] == ([] if host_patch else [visual])
    assert bool(getattr(visual, "_rpu_vision_merger_on_device", False)) is not host_patch
    assert not getattr(visual, "_rpu_vision_aux_w8a16", False)
    assert not getattr(visual, "_rpu_vision_fused_merger", False)
    assert eager_install_runtime["vision_replay_calls"] == [
        "qwen3vl_vision_set_fast_replay", "qwen3vl_vision_set_preload_replay_skip"]
    if not host_patch:
        assert visual._rpu_patch_embed_w_rpu.dtype == torch.float16
        calls = cpu_transfers["calls"]
        assert owner.to_rpu() is model
        assert cpu_transfers["calls"] == calls
    owner.close()
    assert not hasattr(visual, "_rpu_patch_embed_w_rpu")


@pytest.mark.parametrize("factory", [_model_2b, _model_4b])
def test_awq_install_skips_generic_swizzle_binds_scales_and_uses_fp16_aux(
    eager_install_runtime, cpu_transfers, monkeypatch, factory,
):
    from rpu_backend.quant import load

    model = factory()
    model.config.quantization_config = {"quant_method": "compressed-tensors"}
    model.config.tie_word_embeddings = factory is _model_2b
    # The real loader tests cover the sealed full inventory. This double has
    # no decoder layers and exercises the adapter's caller/consumer boundary.
    monkeypatch.setattr(load, "is_qwen3_vl_awq_config", lambda config: config is model.config)
    validations, moves = [], []
    monkeypatch.setattr(load, "_validate_qwen3_vl_awq_model",
                        lambda value: validations.append(value) or True)
    def move(value):
        assert value._rpu_swizzle_started and eager_install_runtime["allocator_enabled"]
        moves.append(value)
    monkeypatch.setattr(load, "_move_materialized_awq_decoder_for_rpu", move)
    monkeypatch.setattr(qvl, "convert_linear_weights_inplace",
                        lambda *a, **k: pytest.fail("packed AWQ was swizzled twice"))
    monkeypatch.setattr(qvl, "_qwen3_vl_text_projection_weights_are_fp16", lambda value: False)
    eager_install_runtime["prepare_before_head"] = True
    owner = qvl.Qwen3VLAdapter(model)
    assert owner.to_rpu() is model
    assert moves == [model] and len(validations) == 3
    assert eager_install_runtime["text_kwargs"]["scale_lists"] == tuple([] for _ in range(7))
    assert not eager_install_runtime["text_kwargs"].get("runtime_align_w8a16", False)
    assert eager_install_runtime["vision_kwargs"]["_pack_fp16_block_weights"] is True
    assert not eager_install_runtime["vision_kwargs"].get("w8a16", False)
    assert eager_install_runtime["eager_calls"] == [model.model.visual]
    assert not getattr(model.model.visual, "_rpu_vision_aux_w8a16", False)
    assert eager_install_runtime["vision_replay_calls"] == [
        "qwen3vl_vision_set_fast_replay", "qwen3vl_vision_set_preload_replay_skip"]
    # AWQ text retains ACC16; refresh the prepared decode plan after the final
    # accumulation policy and lm_head setters even when that policy is false.
    # Its eager-merger Vision sizes AUTO from live SPM, so Text Persistent is
    # reserved cold last.
    assert eager_install_runtime["events"][-4:] == [
        ("set_linear_acc32", 22, False),
        ("set_lm_head", 22),
        ("resolve_decode", False, True),
        ("prepare_text_persistent", 22, 16, 0, True),
    ]
    text = model.model.language_model
    assert text._rpu_decode_stage_descriptor != eager_install_runtime["initial_decode_descriptor"]
    assert text._rpu_decode_stage_descriptor == tuple(
        text._decoder_decode_plan[1].selected.stage_tuple.physical_descriptor)
    events = list(eager_install_runtime["events"])
    assert owner.to_rpu() is model and moves == [model]
    assert eager_install_runtime["events"] == events
    owner.close()


def test_reduced_text_topology_puts_fp16_vision_aux_on_four_lanes(
    eager_install_runtime, cpu_transfers, monkeypatch,
):
    # This tiny installer double isolates the routing decision. The real
    # reduced weight and Vision geometry validators have separate coverage.
    model = _model_2b()
    head = torch.nn.Linear(2, 2, bias=False)
    head.to = lambda *args, **kwargs: head
    model.lm_head = head
    topology = SimpleNamespace(num_cores=4, attn_tp=4, mlp_tp=4, lm_head_tp=4)
    monkeypatch.setattr(qvl, "_qwen3_vl_core_topology", lambda *args: topology)
    monkeypatch.setattr(qvl, "qwen3_vl_text_core_profile", lambda *args: ("2b", 4))
    monkeypatch.setattr(qvl, "_validate_qwen3_vl_reduced_weights", lambda *args: None)
    monkeypatch.setattr(qvl, "pad_decoder_mlp_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(qvl, "_qwen3_vl_text_projection_weights_are_fp16", lambda *args: True)

    adapter = qvl.Qwen3VLAdapter(model)
    assert qvl._is_qwen3_vl_dense_eager_profile(model)
    selected = []
    monkeypatch.setattr(vision, "tp_col_swizzle_mc_weight",
                        lambda value, cores, **kwargs: selected.append(cores) or value.clone())
    adapter.to_rpu()
    visual = model.model.visual
    assert eager_install_runtime["eager_calls"] == [visual]
    assert eager_install_runtime["eager_kwargs"] == [{"cores": 4}]
    assert visual._rpu_vision_merger_on_device and visual._rpu_patch_embed_on_device
    assert visual._rpu_vision_aux_num_cores == 4 and not visual._rpu_vision_aux_w8a16
    assert selected == [4] * 5
    assert eager_install_runtime["fused_calls"] == []
    assert eager_install_runtime["vision_kwargs"]["num_cores"] == 4
    assert not eager_install_runtime["vision_kwargs"].get("_pack_fp16_block_weights", False)
    assert eager_install_runtime["vision_replay_calls"] == []
    assert eager_install_runtime["text_kwargs"]["topology"] is topology
    adapter.close()


@pytest.mark.parametrize("profile", [
    "2b_unknown_semantics", "4b_unknown_semantics",
    "2b_root_quant_metadata", "2b_text_quant_metadata", "2b_vision_quant_metadata",
    "4b_root_quant_metadata", "4b_text_quant_metadata", "4b_vision_quant_metadata",
    "2b_quantized_text", "4b_quantized_text", "8b_quantized_text",
    "2b_cpu_fp32", "4b_cpu_fp32", "8b_cpu_fp32",
    "2b_w4", "4b_w4", "8b_w4",
])
def test_public_eager_default_preserves_other_profiles_and_explicit_fp32(
    eager_install_runtime, cpu_transfers, monkeypatch, profile,
):
    model = _model_2b() if profile.startswith("2b") else (_model_4b() if profile.startswith("4b") else _model_8b())
    if profile.endswith("unknown_semantics"):
        model.config.text_config.rms_norm_eps = 1e-5
    if "_quant_metadata" in profile:
        scope = profile.split("_", 2)[1]
        owner = {
            "root": model.config,
            "text": model.config.text_config,
            "vision": model.config.vision_config,
        }[scope]
        owner.quantization_config = {}
    if profile.endswith("cpu_fp32"):
        monkeypatch.setenv("RPU_QWEN3VL_VISION_HOST_FP32_PATCH", "1")
    if profile.endswith("quantized_text"):
        monkeypatch.setattr(qvl, "_qwen3_vl_text_projection_weights_are_fp16", lambda model: False)
    if profile.endswith("w4"):
        monkeypatch.setattr(qvl, "_lm_head_optimization_config", lambda model: (True, 0, False, 32))
        monkeypatch.setattr(qvl, "_quantize_pack_lm_head_w4_groupwise",
                            lambda *args, **kwargs: (torch.ones(2, dtype=torch.uint8), torch.ones(2).half()))
        def set_lm_head(*args):
            eager_install_runtime["lm_head_set"] = True
        monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_lm_head", set_lm_head, raising=False)
    adapter = qvl.Qwen3VLAdapter(model)
    adapter.to_rpu()
    assert eager_install_runtime["eager_calls"] == []
    # W4 transfers its lm-head tensors; none of these paths convert vision aux.
    if not profile.endswith("w4"):
        assert cpu_transfers["calls"] == 0
    assert not hasattr(model.model.visual, "_rpu_vision_merger_on_device")
    assert not hasattr(model.model.visual.merger, "_rpu_fc1_w")
    if profile.endswith("cpu_fp32"):
        assert model.model.visual._rpu_vision_host_fp32_patch
    assert eager_install_runtime["vision_replay_calls"] == (
        ["qwen3vl_vision_set_fast_replay", "qwen3vl_vision_set_preload_replay_skip"]
        if profile in ("2b_cpu_fp32", "4b_cpu_fp32") else [])
    adapter.close()


@pytest.mark.parametrize("scope", ["root", "text", "vision"])
@pytest.mark.parametrize("attribute", ["quant_config", "quantization_config"])
def test_public_8b_rejects_nested_quant_metadata_before_claim_or_transfer(
    eager_install_runtime, cpu_transfers, scope, attribute,
):
    model = _model_8b()
    owner = {
        "root": model.config,
        "text": model.config.text_config,
        "vision": model.config.vision_config,
    }[scope]
    setattr(owner, attribute, {})

    with pytest.raises(
        qvl.UnsupportedModelError,
        match="requires an FP16 checkpoint or a registered runtime quantization profile",
    ):
        qvl.Qwen3VLAdapter(model).to_rpu()

    assert eager_install_runtime["events"] == []
    assert eager_install_runtime["eager_calls"] == []
    assert cpu_transfers["calls"] == 0


@pytest.mark.parametrize("failure", ["aux_transfer", "later_lm_head"])
@pytest.mark.parametrize("factory", [_model_2b, _model_4b, _model_8b])
def test_public_eager_install_failure_retires_handles_and_aux(
    eager_install_runtime, cpu_transfers, failure, factory,
):
    model = factory()
    if failure == "aux_transfer":
        cpu_transfers["fail_at"] = 6
        match = "transfer interrupted"
    else:
        eager_install_runtime["set_error"] = RuntimeError("lm-head registration failed")
        match = "lm-head registration failed"
    adapter = qvl.Qwen3VLAdapter(model)
    with pytest.raises(RuntimeError, match=match):
        adapter.to_rpu()
    visual = model.model.visual
    assert eager_install_runtime["eager_calls"] == [visual]
    assert ("destroy", "vision") in eager_install_runtime["events"]
    assert ("destroy", "text") in eager_install_runtime["events"]
    assert not adapter._rpu_is_ready
    assert not hasattr(visual, "_rpu_patch_embed_w_rpu")
    assert all(not any(name in vars(merger) for name in vision._VISION_MERGER_AUX_ATTRS)
               for merger in (visual.merger, *visual.deepstack_merger_list))
    transfers = cpu_transfers["calls"]
    with pytest.raises(qvl.RPUBackendError, match="prior swizzle attempt|closed"):
        adapter.to_rpu()
    assert cpu_transfers["calls"] == transfers


@pytest.mark.parametrize("field,value", [
    ("hidden_size", 1024), ("intermediate_size", 4352), ("depth", 26),
    ("num_heads", 8), ("patch_size", 14), ("temporal_patch_size", 1),
    ("spatial_merge_size", 4), ("deepstack_visual_indexes", [5, 11, 17]),
    ("hidden_act", "gelu"), ("out_hidden_size", 2048),
    ("model_type", "other"), ("in_channels", 1), ("num_position_embeddings", 1024),
])
def test_8b_vision_mismatch_is_rejected_before_default_aux_or_weight_mutation(
    eager_install_runtime, cpu_transfers, field, value,
):
    model = _model_8b()
    setattr(model.config.vision_config, field, value)
    with pytest.raises(qvl.UnsupportedModelError):
        qvl.Qwen3VLAdapter(model).to_rpu()
    assert eager_install_runtime["claims"] == []
    assert eager_install_runtime["events"] == []
    assert eager_install_runtime["eager_calls"] == []
    assert cpu_transfers["calls"] == 0
    assert not hasattr(model, "_rpu_swizzle_started")


def test_existing_2b_delivery_keeps_its_separate_fused_replay_install(install_runtime, monkeypatch):
    # Delivery config/weight admission is outside this tiny routing double.
    # Exercise the real installation branch after those validators accept it.
    monkeypatch.setattr(qvl, "_validate_delivery_weights", lambda model: True)
    monkeypatch.setattr(qvl, "_normalize_delivery_execution", lambda value: value)
    monkeypatch.setattr(qvl, "_qwen3_vl_text_scale_lists", lambda model: {})
    monkeypatch.setattr(qvl, "_text_decode_execution_plan", lambda *args, **kwargs:
                        SimpleNamespace(selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(1,)))))
    calls = []
    monkeypatch.setattr(qvl, "register_qwen3vl_vision_fused_merger", lambda visual: calls.append("fused"))
    enable = qvl.enable_qwen3_vl_vision_merger_on_device
    def eager(visual):
        calls.append("eager")
        enable(visual)
    monkeypatch.setattr(qvl, "enable_qwen3_vl_vision_merger_on_device", eager)
    for name in (
        "causal_decoder_set_chunk_envelope", "qwen3vl_vision_set_delivery_compatibility",
        "qwen3vl_vision_set_fast_replay", "qwen3vl_vision_set_preload_replay_skip",
        "qwen3vl_vision_set_bake_merger", "causal_decoder_set_fast_replay",
        "causal_decoder_set_preload_replay_skip",
    ):
        monkeypatch.setattr(torch.ops.rpu, name,
                            lambda *args, name=name: calls.append(name), raising=False)
    delivery_model = _Model()
    delivery_model.model.language_model._rpu_mrope_capacity = 576
    adapter = qvl.Qwen3VLAdapter(delivery_model)
    model = adapter.to_rpu()
    assert calls == [
        "causal_decoder_set_chunk_envelope", "qwen3vl_vision_set_delivery_compatibility",
        "fused", "eager", "qwen3vl_vision_set_fast_replay",
        "qwen3vl_vision_set_preload_replay_skip", "qwen3vl_vision_set_bake_merger",
        "causal_decoder_set_fast_replay", "causal_decoder_set_preload_replay_skip",
    ]
    assert model.model.language_model._rpu_text_fast_replay
    assert model.model.language_model._rpu_text_fused_decode_position
    assert adapter._text_fast_replay_policy is True
    # The delivery owns forced-true replay even though its fixed config omits
    # the public opt-in. An otherwise legal reconfigure must retain that policy.
    from rpu_backend.api._execution import reconfigure_rpu_execution
    monkeypatch.setattr(qvl, "_delivery_profile_active", lambda model: True)
    reconfigure_rpu_execution(model, {"vision": {"chunk_size": 144}})
    assert adapter._text_fast_replay_policy is True
    assert calls.count("causal_decoder_set_fast_replay") == 1
    adapter.close()


@pytest.mark.parametrize("profile", ["4b_w8", "4b_fp16", "32b_w8"])
def test_ordinary_vision_skip_preserves_precision_without_bake_or_text_skip(
    install_runtime, monkeypatch, profile,
):
    from rpu_backend.quant.convert_qwen3_vl import QUANT_CONFIG
    from rpu_backend.tests._w8a16_staged_handoff import _qwen3_vl_32b_w8_config

    model = _model_4b()
    if profile == "4b_w8":
        model.config.tie_word_embeddings = False
        model.config.text_config.tie_word_embeddings = False
        model.config.text_config.max_position_embeddings = 262144
        model.config.quant_config = dict(QUANT_CONFIG)
        model.lm_head.weight = torch.nn.Parameter(torch.ones(2,2,dtype=torch.int8), requires_grad=False)
        model.lm_head.weight_scale = torch.ones(2,dtype=torch.float16)
        assert qvl._is_qwen3_vl_4b_w8a16_config(model.config)
    elif profile == "32b_w8":
        model.config = _qwen3_vl_32b_w8_config()
        assert qvl._is_qwen3_vl_32b_w8a16_config(model.config)

    # Keep the real config predicates and top-level installation flow. Full
    # checkpoint inventories/quantization have separate loader coverage; this
    # fixture replaces only their large tensors and native device transport.
    monkeypatch.setattr(qvl, "_validate_qwen3_vl_4b_w8a16_model",
                        lambda value: qvl._is_qwen3_vl_4b_w8a16_config(value.config))
    monkeypatch.setattr(qvl, "_validate_qwen3_vl_32b_w8a16_model",
                        lambda value: qvl._is_qwen3_vl_32b_w8a16_config(value.config))
    monkeypatch.setattr(qvl, "_qwen3_vl_text_scale_lists", lambda value: ([],)*7)
    monkeypatch.setattr(qvl, "_qwen3_vl_text_projection_weights_are_fp16",
                        lambda value: profile == "4b_fp16")
    from rpu_backend.quant import load as quant_load
    bank_moves = []
    monkeypatch.setattr(quant_load, "_move_materialized_w8a16_decoder_for_rpu",
                        lambda value: bank_moves.append(value))
    prepared_policies = []

    def prepare_arenas(graph_policy):
        assert profile == "32b_w8"
        assert graph_policy.graph_arena_count > 0
        assert install_runtime["events"][-2:] == [
            ("claim", "model"), ("set_caching_allocator", True),
        ]
        prepared_policies.append(graph_policy)
        install_runtime["events"].append(("prepare_arenas", "model"))
        return True

    from rpu_backend.graph import GraphRuntimePolicy
    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", prepare_arenas)
    generic_bank_moves = []

    def move_decoder(value, *, dtype, per_layer=False):
        assert value is model and value._rpu_swizzle_started is True
        assert install_runtime["allocator_enabled"] is True
        if profile == "32b_w8":
            assert prepared_policies
            assert install_runtime["events"][-1] == ("prepare_arenas", "model")
        generic_bank_moves.append((value, dtype, per_layer))

    monkeypatch.setattr(
        quant_load, "_move_materialized_decoder_for_rpu",
        move_decoder,
    )
    native_head = torch.ops.rpu.causal_decoder_set_lm_head
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_lm_head",
                        lambda handle, weight, *scale: native_head(handle, weight))
    calls = []

    def cold_event(name, visual=None, *, w8a16=False):
        visual = model.model.visual if visual is None else visual
        owner = model._qwen3_vl_retirement_owner
        assert not owner._rpu_is_ready
        assert not getattr(visual, "_rpu_vision_has_dispatched", False)
        assert not visual._rpu_vision_graph_cache.is_frozen()
        assert ("install", "text") in install_runtime["events"]
        calls.append((name, w8a16))

    monkeypatch.setattr(qvl, "enable_qwen3_vl_vision_merger_on_device",
                        lambda visual, **kwargs: cold_event("eager", visual, **kwargs))
    monkeypatch.setattr(qvl, "register_qwen3vl_vision_fused_merger",
                        lambda visual, **kwargs: cold_event("fused", visual, **kwargs))
    for name in ("qwen3vl_vision_set_n1200_tm160", "qwen3vl_vision_set_fast_replay", "qwen3vl_vision_set_preload_replay_skip", "qwen3vl_vision_set_fp16_chunked_merger", "qwen3vl_vision_set_fp16_compact_encoder", "qwen3vl_vision_set_w8_compact_encoder"):
        def set_vision(handle, enabled, name=name):
            assert handle == 11 and enabled is True
            assert getattr(model.model.visual, "_rpu_vision_chunked_merger_w8a16", False) is (profile == "4b_w8")
            cold_event(name)
        monkeypatch.setattr(torch.ops.rpu, name, set_vision, raising=False)
    for name in ("qwen3vl_vision_set_bake_merger", "causal_decoder_set_fast_replay",
                 "causal_decoder_set_preload_replay_skip"):
        monkeypatch.setattr(torch.ops.rpu, name,
                            lambda *args: pytest.fail("ordinary profiles must retain post_fn and text op emission"),
                            raising=False)

    instance = qvl.Qwen3VLAdapter(model)
    assert instance.to_rpu() is model
    expected = ([('eager', True), ('fused', True),
                 ('qwen3vl_vision_set_n1200_tm160', False),
                 ('qwen3vl_vision_set_w8_compact_encoder', False),
                 ('qwen3vl_vision_set_fast_replay', False),
                 ('qwen3vl_vision_set_preload_replay_skip', False)] if profile == "4b_w8"
                else [('qwen3vl_vision_set_fast_replay', False),
                      ('qwen3vl_vision_set_preload_replay_skip', False),
                      ('eager', False)] if profile == "4b_fp16" else [])
    assert calls == expected
    assert getattr(model.model.visual, "_rpu_vision_chunked_merger_w8a16", False) is (profile == "4b_w8")
    assert getattr(model.model.visual, "_rpu_vision_compact_encoder_w8a16", False) is (profile == "4b_w8")
    assert not getattr(model.model.visual, "_rpu_vision_chunked_merger_fp16", False)
    assert ("set_vision_linear_acc32", 11, False) in install_runtime["events"]
    assert ("set_linear_acc32", 22, False) in install_runtime["events"]
    assert not model.model.language_model._rpu_text_fused_decode_position
    assert not getattr(model.model.language_model, "_rpu_text_fast_replay", False)
    assert instance.to_rpu() is model
    assert calls == expected
    assert bank_moves == ([model] if profile == "4b_w8" else [])
    assert generic_bank_moves == (
        [(model, torch.float16, True)] if profile == "4b_fp16" else
        [(model, torch.int8, True)] if profile == "32b_w8" else [])
    if profile == "32b_w8":
        assert len(prepared_policies) == 1
        assert instance._graph_arena_policy is prepared_policies[0]
    else:
        assert prepared_policies == []
    instance.close()
    assert not hasattr(model.model.visual, "_rpu_vision_compact_encoder_w8a16")


@pytest.fixture
def n1200_merged_snapshot_owner(cpu_transfers, monkeypatch):
    """Real Python grouping with four native slots overwritten by every image."""
    from contextlib import contextmanager

    class Visual(SimpleNamespace):
        def fast_pos_embed_interpolate(self, grid):
            return torch.zeros(int(grid.prod(-1).sum()), 2, dtype=torch.float16)

    def make(profile="w8"):
        signatures, state = [], {"fault": None}
        slots = [torch.empty(300, 3, dtype=torch.float16) for _ in range(4)]
        raw = torch.empty(1, 1200, 2, dtype=torch.float16)

        @contextmanager
        def capture(signature):
            signatures.append(signature)
            yield
            # Native BUILD/REPLAY completes on capture exit. A snapshot taken
            # inside capture would preserve stale data from the preceding image.
            raw.copy_(state.pop("pending_raw"))

        visual = Visual(
            config=SimpleNamespace(out_hidden_size=3),
            _rpu_vision_handle=11, _rpu_vision_spatial_merge_size=2,
            _rpu_vision_num_layers=3, _rpu_vision_hidden_size=2,
            _rpu_vision_kv_cache=SimpleNamespace(k_caches=[None] * 3, v_caches=[None] * 3,
                max_seq_len=1200, reset_to_position=lambda position: None),
            _rpu_vision_graph_cache=SimpleNamespace(capture=capture),
            _rpu_vision_deepstack_hash=3, _rpu_vision_max_hw=40,
            _rpu_vision_host_fp32_patch=False, _rpu_vision_patch_embed_w=torch.eye(2).half(),
            _rpu_vision_patch_embed_b=None,
            _rpu_vision_position_idx_keepalive=torch.zeros(1200, 2).short(),
            _rpu_vision_deepstack_indexes=[0, 1, 2],
            _rpu_vision_fused_merger=True, _rpu_vision_merger_on_device=True,
            _rpu_vision_fused_merger_w8a16=profile == "w8",
            _rpu_vision_chunked_merger_w8a16=profile == "w8",
            _rpu_vision_compact_encoder_w8a16=profile == "w8",
            _rpu_vision_chunked_merger_fp16=profile == "fp16",
            _rpu_vision_compact_encoder_fp16=profile == "fp16",
            _rpu_vision_batch_enabled=False, _rpu_vision_batch_cap=1,
            merger=SimpleNamespace(linear_fc2=SimpleNamespace(out_features=3)),
            deepstack_merger_list=[object(), object(), object()],
        )

        def plan(handle, patches, batch, *args, **kwargs):
            assert patches == 1200 and batch == 1
            descriptor = (patches, 1200)
            return SimpleNamespace(selected=SimpleNamespace(execution_len=patches,
                stage_tuple=SimpleNamespace(compute_chunk=1200, physical_descriptor=descriptor)),
                graph_key_words=lambda: descriptor, graph_mode="GRAPH_COMPOSITE_CHILD",
                selection_scope="TEST", physical_plan_digest="test", plan_digest="test")

        def forward(handle, values, k, v, patches, batch, descriptor):
            assert patches == 1200 and batch == 1
            state["pending_raw"] = values.clone()
            for index, slot in enumerate(slots):
                slot.copy_(values[0, ::4, :1] + torch.arange(3).half() / 8 + 10 * (index + 1))
            if state["fault"] == "raw_rows":
                return raw[:, :1]
            if state["fault"] == "raw_dtype":
                return raw.float()
            return raw

        def pop(handle):
            values = list(slots)
            fault = state["fault"]
            if fault == "rank":
                values[2] = values[2].unsqueeze(0)
            elif fault == "rows":
                values[2] = values[2][:1]
            elif fault == "width":
                values[2] = values[2][:, :1]
            elif fault == "dtype":
                values[2] = values[2].float()
            return values

        monkeypatch.setattr(vision, "_plan_qwen3_vl_vision_execution", plan)
        monkeypatch.setattr(vision, "_deepstack_output_cls", lambda: SimpleNamespace)
        monkeypatch.setattr(vision.rpu_backend.graph, "GraphSignature", lambda **kw: SimpleNamespace(**kw))
        monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_forward", forward, raising=False)
        monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_get_resolved_chunk_size", lambda handle: 1200, raising=False)
        monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_pop_merged", pop, raising=False)
        monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)

        def call(order, *, keep_fp16=True):
            pixels = torch.cat([torch.full((1200, 2), float(code)).half() for code in order])
            grid = torch.tensor([[1, 30, 40]] * len(order))
            return vision._rpu_vision_forward(visual, pixels, grid, _rpu_keep_merged_fp16=keep_fp16)

        def expected(order):
            return [torch.cat([torch.full((300, 3), float(code)).half() +
                torch.arange(3).half() / 8 + 10 * (index + 1) for code in order]) for index in range(4)]

        return SimpleNamespace(call=call, expected=expected, slots=slots, raw=raw,
                               state=state, signatures=signatures)

    return make


@pytest.mark.parametrize("profile", ["w8", "fp16"])
def test_n1200_multi_group_snapshots_preserve_order_and_retained_owners(n1200_merged_snapshot_owner, profile):
    owner = n1200_merged_snapshot_owner(profile)
    retained = []
    for order in ((1, 2), (2, 1), (1, 2)):
        output = owner.call(order)
        values = [output.pooler_output, *output.deepstack_features]
        expected = owner.expected(order)
        assert len(values) == 4
        assert all(value.device.type == "cpu" and value.dtype == torch.float16
                   and value.is_contiguous() and value.shape == (600, 3) for value in values)
        assert all(torch.equal(value, ref) for value, ref in zip(values, expected))
        assert len({value.untyped_storage().data_ptr() for value in values}) == 4
        assert not ({value.untyped_storage().data_ptr() for value in values}
                    & {slot.untyped_storage().data_ptr() for slot in owner.slots})
        raw_expected = torch.cat([torch.full((1200, 2), float(code)).half() for code in order])
        raw = output.last_hidden_state
        assert raw.dtype == torch.float16 and raw.is_contiguous()
        assert raw.device == owner.raw.device and torch.equal(raw, raw_expected)
        assert raw.untyped_storage().data_ptr() != owner.raw.untyped_storage().data_ptr()
        values.append(raw)
        expected.append(raw_expected)
        for old_values, old_expected in retained:
            assert all(torch.equal(value, ref) for value, ref in zip(old_values, old_expected))
            assert not ({value.untyped_storage().data_ptr() for value in values}
                        & {value.untyped_storage().data_ptr() for value in old_values})
        retained.append((values, expected))
    assert owner.signatures[:2] == owner.signatures[2:4] == owner.signatures[4:6]


@pytest.mark.parametrize("profile", ["w8", "fp16"])
@pytest.mark.parametrize("order,keep_fp16", [((1,), True), ((1, 2), False)])
def test_n1200_snapshot_single_group_and_fp32_fallback(n1200_merged_snapshot_owner, profile, order, keep_fp16):
    owner = n1200_merged_snapshot_owner(profile)
    output = owner.call(order, keep_fp16=keep_fp16)
    values = [output.pooler_output, *output.deepstack_features]
    expected = owner.expected(order)
    raw = output.last_hidden_state
    raw_expected = torch.cat([torch.full((1200, 2), float(code)).half() for code in order])
    assert raw.dtype == torch.float16 and torch.equal(raw, raw_expected)
    dtype = torch.float16 if keep_fp16 else torch.float32
    assert all(value.device.type == "cpu" and value.dtype == dtype for value in values)
    assert all(torch.equal(value, ref) for value, ref in zip(values, expected))
    owner.call((3, 4), keep_fp16=not keep_fp16)
    assert all(torch.equal(value, ref) for value, ref in zip(values, expected))
    assert torch.equal(raw, raw_expected)


@pytest.mark.parametrize("fault", ["rank", "rows", "width", "dtype", "raw_rows", "raw_dtype"])
def test_n1200_snapshot_rejects_broadcastable_or_wrong_dtype_deepstack(n1200_merged_snapshot_owner, fault):
    owner = n1200_merged_snapshot_owner()
    output = owner.call((1, 2))
    retained = [output.pooler_output, *output.deepstack_features, output.last_hidden_state]
    expected = owner.expected((1, 2)) + [torch.cat([
        torch.full((1200, 2), float(code)).half() for code in (1, 2)])]
    owner.state["fault"] = fault
    # copy_ would silently broadcast one row/column and convert FP32 to FP16.
    # Native return metadata must be checked before accepting these outputs.
    with pytest.raises(RuntimeError):
        owner.call((3, 4))
    assert all(torch.equal(value, ref) for value, ref in zip(retained, expected))


@pytest.mark.parametrize("acc32", [False,True])
def test_four_core_w8_auxiliary_install_and_patch_dispatch_bind_same_cold_layout(cpu_transfers,monkeypatch,acc32):
    visual=_visual();visual._rpu_vision_num_cores=4;visual._rpu_vision_linear_acc32=acc32
    selected=[]
    monkeypatch.setattr(vision,"tp_col_swizzle_mc_weight",
        lambda weight,cores,**kwargs:selected.append(cores) or weight.clone())
    vision.enable_qwen3_vl_vision_merger_on_device(visual,cores=4,w8a16=True)
    assert selected==[4]*5
    assert visual._rpu_vision_aux_num_cores==4
    assert all(m._rpu_merger_num_cores==4 for m in (visual.merger,*visual.deepstack_merger_list))
    calls=[]
    def linear(x,w,b,partition,cores,enabled,scale):
        assert (partition,cores,enabled)==(1,4,acc32)
        calls.append((w,scale))
        return torch.nn.functional.linear(x.float(),w.float()*scale.float()[:,None],b.float()).half()
    monkeypatch.setattr(torch.ops.rpu,"linear_with_accumulation",linear,raising=False)
    monkeypatch.setattr(torch.ops.rpu,"linear_w8a16_into",lambda *a:pytest.fail("entered TP8-only patch path"),raising=False)
    x=torch.arange(8,dtype=torch.float16).reshape(4,2)
    expected=torch.nn.functional.linear(x.float(),
        visual._rpu_patch_embed_w_rpu.float()*visual._rpu_vision_patch_embed_scale.float()[:,None],
        visual._rpu_patch_embed_b_rpu.float()).half()
    assert torch.equal(vision._vision_patch_linear(visual,x),expected)
    assert calls==[(visual._rpu_patch_embed_w_rpu,visual._rpu_vision_patch_embed_scale)]
    visual._rpu_vision_aux_num_cores=8
    with pytest.raises(ValueError,match="bound four-core layout"):
        vision._vision_patch_linear(visual,x)
    with pytest.raises(ValueError,match="core layout"):
        vision.enable_qwen3_vl_vision_merger_on_device(visual,cores=4,w8a16=True)


@pytest.mark.parametrize("fail_at", [1,7,14])
def test_four_core_w8_auxiliary_rollback_removes_all_core_stamps(cpu_transfers,fail_at):
    visual=_visual();visual._rpu_vision_num_cores=4
    cpu_transfers["fail_at"]=fail_at
    with pytest.raises(RuntimeError,match="transfer interrupted"):
        vision.enable_qwen3_vl_vision_merger_on_device(visual,cores=4,w8a16=True)
    assert not hasattr(visual,"_rpu_vision_aux_num_cores")
    assert all(not any(name in vars(m) for name in vision._VISION_MERGER_AUX_ATTRS)
        for m in (visual.merger,*visual.deepstack_merger_list))


@pytest.mark.parametrize("patch_embed", [True, False])
def test_four_core_fp16_auxiliaries_bind_four_lanes_without_scales(cpu_transfers, monkeypatch, patch_embed):
    visual = _visual(); visual._rpu_vision_num_cores = 4
    selected = []
    monkeypatch.setattr(vision, "tp_col_swizzle_mc_weight",
                        lambda weight, cores, **kwargs: selected.append((cores, kwargs)) or weight.clone())
    vision.enable_qwen3_vl_vision_merger_on_device(visual, cores=4, patch_embed=patch_embed)
    assert selected == [(4, {})] * (5 if patch_embed else 4)
    assert visual._rpu_vision_merger_on_device and not visual._rpu_vision_aux_w8a16
    assert visual._rpu_vision_aux_num_cores == 4
    assert getattr(visual, "_rpu_patch_embed_on_device", False) is patch_embed
    mergers = (visual.merger, *visual.deepstack_merger_list)
    assert all(m._rpu_merger_num_cores == 4 and m._rpu_fc1_scale is None for m in mergers)
    calls = []
    def linear(x, w, b, partition, cores, enabled, scale, gelu_erf=False):
        assert (partition, cores, enabled, scale) == (1, 4, False, None)
        assert gelu_erf is (w is visual.merger._rpu_fc1_w)
        calls.append(w)
        return torch.nn.functional.linear(x.float(), w.float(), b.float()).half()
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", linear, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "linear_into",
                        lambda *a: pytest.fail("entered TP8-only patch path"), raising=False)
    out = vision._merger_forward_on_device(visual.merger, torch.ones(4, 2))
    assert out.dtype == torch.float32 and calls[-2:] == [visual.merger._rpu_fc1_w, visual.merger._rpu_fc2_w]
    if patch_embed:
        x = torch.arange(8, dtype=torch.float16).reshape(4, 2)
        expected = torch.nn.functional.linear(
            x.float(), visual._rpu_patch_embed_w_rpu.float(), visual._rpu_patch_embed_b_rpu.float()).half()
        assert torch.equal(vision._vision_patch_linear(visual, x), expected)
    calls_before = cpu_transfers["calls"]
    vision.enable_qwen3_vl_vision_merger_on_device(visual, cores=4, patch_embed=patch_embed)
    assert cpu_transfers["calls"] == calls_before
    with pytest.raises(ValueError, match="core layout"):
        vision.enable_qwen3_vl_vision_merger_on_device(visual, cores=4, w8a16=True)


@pytest.mark.parametrize("caller", ["merger", "patch"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_eager_reduced_auxiliaries_suspend_intra_flush_and_restore_it(
        cpu_transfers, monkeypatch, caller, enabled, fail):
    if caller == "merger":
        merger = _merger()
        vision._convert_vision_merger_for_rpu(merger, cores=4)
        run, launches = (lambda: vision._merger_forward_on_device(merger, torch.ones(4, 2))), 2
    else:
        visual = _visual(); visual._rpu_vision_num_cores = 4
        vision.enable_qwen3_vl_vision_merger_on_device(visual, cores=4)
        run, launches = (lambda: vision._vision_patch_linear(visual, torch.ones(4, 2, dtype=torch.float16))), 1
    state = {"flush": enabled, "sets": []}
    monkeypatch.setattr(torch.rpu, "get_ddr_flush", lambda: state["flush"])
    monkeypatch.setattr(torch.rpu, "set_ddr_flush",
                        lambda value: state.update(flush=value) or state["sets"].append(value))
    seen = []
    def linear(x, w, b, partition, cores, acc32, scale, gelu_erf=False):
        seen.append(state["flush"])
        if fail and len(seen) == launches:
            raise RuntimeError("launch failed")
        return torch.nn.functional.linear(x.float(), w.float(), b.float()).half()
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", linear, raising=False)
    if fail:
        with pytest.raises(RuntimeError, match="launch failed"):
            run()
    else:
        run()
    assert seen == [False] * launches
    assert state["flush"] is enabled
    assert state["sets"] == ([False, True] if enabled else [])
