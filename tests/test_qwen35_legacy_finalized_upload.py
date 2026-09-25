"""Host checks for legacy finalized-only installation; no hardware or model weights."""
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters.qwen3_5 import cores, text
from rpu_backend.adapters.qwen3_5.quant_scope import legacy_w8a16_quant_config
from rpu_backend.runtime import weights
from rpu_backend.runtime import _native_retirement


class _StopAtNativeWeights(Exception):
    pass


def _fixture():
    """Small real tensors exercise both mixers and the legacy mixed precision.

    This tests storage migration after admission, not admission of this tiny
    geometry as a supported model. Only geometry/profile admission is stubbed.
    """
    torch.manual_seed(401)
    config = NS(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                num_attention_heads=8, num_key_value_heads=8, head_dim=32,
                linear_num_key_heads=8, linear_num_value_heads=16,
                linear_key_head_dim=128, linear_value_head_dim=128,
                linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
                layer_types=["linear_attention", "full_attention"],
                quant_config=legacy_w8a16_quant_config())

    def linear(n, k, quant=True):
        module = torch.nn.Linear(k, n, bias=False, dtype=torch.float16)
        if quant:
            module.weight = torch.nn.Parameter(
                torch.randint(-127, 128, (n, k), dtype=torch.int8),
                requires_grad=False)
            module.register_buffer("weight_scale", torch.rand(n).half())
        return module

    def norm(n):
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.rand(n).half())
        return module

    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList()
    inner.embed_tokens = torch.nn.Embedding(16, 128, dtype=torch.float16)
    inner.norm = norm(128)
    for full in (False, True):
        layer = torch.nn.Module()
        layer.input_layernorm = norm(128)
        layer.post_attention_layernorm = norm(128)
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_proj, layer.mlp.up_proj = linear(256, 128), linear(256, 128)
        layer.mlp.down_proj = linear(128, 256)
        if full:
            attn = layer.self_attn = torch.nn.Module()
            attn.q_proj = linear(512, 128)
            attn.k_proj, attn.v_proj = linear(256, 128), linear(256, 128)
            attn.o_proj = linear(128, 256)
            attn.q_norm, attn.k_norm = norm(32), norm(32)
        else:
            attn = layer.linear_attn = torch.nn.Module()
            attn.in_proj_qkv = linear(4096, 128, False)
            attn.in_proj_z = linear(2048, 128, False)
            attn.in_proj_b, attn.in_proj_a = linear(16, 128, False), linear(16, 128, False)
            attn.out_proj = linear(128, 2048, False)
            attn.conv1d = torch.nn.Conv1d(4096, 4096, 4, groups=4096,
                                        bias=False, dtype=torch.float16)
            attn.norm = norm(128)
            attn.A_log = torch.nn.Parameter(torch.rand(16).half())
            attn.dt_bias = torch.nn.Parameter(torch.rand(16).half())
        inner.layers.append(layer)
    return inner, config


@pytest.fixture
def host_device(monkeypatch):
    # Module.to still uses its real recursive _apply path, with independent
    # CPU storage standing in for each device allocation. All swizzles and
    # scale transformations are production implementations.
    original_to = torch.Tensor.to

    def tensor_to(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if isinstance(device, (str, torch.device)) and torch.device(device).type == "rpu":
            if args:
                args = (torch.device("cpu"), *args[1:])
            else:
                kwargs["device"] = torch.device("cpu")
            kwargs["copy"] = True
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", tensor_to)
    monkeypatch.setattr(cores, "validate_core_profile", lambda *a: cores.core_topology())
    monkeypatch.setattr(text, "qwen3_5_kv_replication", lambda *a: 1)
    monkeypatch.setattr(weights, "_release_cpu_weight_pages", lambda: None)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_create", lambda: 71, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_destroy", lambda *a: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_linear_acc32", lambda *a: None, raising=False)

    class Resource:
        def __init__(self, *args, **kwargs):
            self.keepalive = kwargs["keepalive"]
            self.finalizer = None
            self.handle = None

    monkeypatch.setattr(_native_retirement, "_InstalledNativeResource", Resource)


def _install(inner, config):
    return text._install_qwen3_5_text_for_rpu_impl(
        inner, config, max_seq_len=16, chunk_size_cap=0, exact_chunk_size=0,
        padding_budget=64, padding_rows="auto", execution_config={},
        control_snapshot=(), graph_cache=None, prefill_graph=None,
        cpu_stage_weights=False, pending_state=text._Qwen35NativeState())


@pytest.mark.parametrize("free_sources", [True, False])
def test_finalized_upload_preserves_payload_and_source_policy(
    monkeypatch, host_device, free_sources,
):
    monkeypatch.setenv("RPU_QWEN3_5_FREE_HF_WEIGHTS", str(int(free_sources)))
    payloads = []

    def capture(*args):
        def snapshot(value):
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
            if isinstance(value, list):
                return [snapshot(x) for x in value]
            return value
        payloads.append([snapshot(x) for x in args])
        raise _StopAtNativeWeights

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", capture, raising=False)
    # First reproduce the prior whole-stack move, then upload only final projections.
    for bounded in (False, True):
        inner, config = _fixture()
        monkeypatch.setattr(text, "validate_legacy_text_profile", lambda c: bounded)
        moves = []
        original_to = torch.nn.Module.to

        def module_to(module, *args, **kwargs):
            if bounded:
                assert module is not inner and module not in inner.layers
                assert not isinstance(module, (torch.nn.Linear, torch.nn.Conv1d))
                moves.append(module)
            return original_to(module, *args, **kwargs)

        raw_projection_pointers = {
            module.weight.data_ptr() for module in inner.modules()
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d))}
        original_tensor_to = torch.Tensor.to

        def finalized_to(tensor, *args, **kwargs):
            device = kwargs.get("device", args[0] if args else None)
            if (bounded and isinstance(device, (str, torch.device))
                    and torch.device(device).type == "rpu"):
                assert tensor.data_ptr() not in raw_projection_pointers
            return original_tensor_to(tensor, *args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(torch.nn.Module, "to", module_to)
            scoped.setattr(torch.Tensor, "to", finalized_to)
            if not bounded:
                inner.to("rpu")
            with pytest.raises(_StopAtNativeWeights):
                _install(inner, config)
        assert len(moves) == (9 if bounded else 0)
        assert (inner.layers[0].mlp.gate_proj.weight.numel() == 0) == free_sources
        assert (inner.layers[1].self_attn.q_proj.weight_scale.numel() == 0) == free_sources

    def same(left, right):
        if isinstance(left, torch.Tensor):
            assert left.dtype == right.dtype and torch.equal(left, right)
        elif isinstance(left, list):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                same(a, b)
        else:
            assert left == right
    same(*payloads)


def test_invalid_last_scale_rejects_before_any_bounded_move(monkeypatch, host_device):
    inner, config = _fixture()
    inner.layers[-1].self_attn.o_proj.weight_scale = torch.ones(1, dtype=torch.float16)
    monkeypatch.setattr(text, "validate_legacy_text_profile", lambda c: True)
    monkeypatch.setattr(torch.nn.Module, "to", lambda *a, **kw: pytest.fail("moved before validation"))
    with pytest.raises(text.RPUBackendError, match="scale numel"):
        _install(inner, config)


def test_move_failure_does_not_upload_later_layers(monkeypatch, host_device):
    inner, config = _fixture()
    monkeypatch.setattr(text, "validate_legacy_text_profile", lambda c: True)
    original_to = torch.nn.Module.to

    def fail(module, *args, **kwargs):
        if module is inner.layers[0].input_layernorm:
            raise RuntimeError("allocation failure")
        if module is inner.layers[1].input_layernorm:
            pytest.fail("later layer uploaded after failure")
        return original_to(module, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "to", fail)
    with pytest.raises(RuntimeError, match="allocation failure"):
        _install(inner, config)
    assert inner.layers[1].self_attn.q_proj.weight.numel() > 0


def test_legacy_adapter_defers_bulk_move_and_preserves_cold_resource_order(monkeypatch):
    from rpu_backend.adapters import qwen3_5 as adapter
    from rpu_backend.graph import GraphRuntimePolicy

    events = []
    config = NS(hidden_size=5120, intermediate_size=17408, num_hidden_layers=64,
                num_attention_heads=24, num_key_value_heads=4, head_dim=256,
                linear_num_key_heads=16, linear_num_value_heads=48,
                linear_key_head_dim=128, linear_value_head_dim=128,
                linear_conv_kernel_dim=4,
                rope_parameters={"rope_type": "default", "rope_theta": 10000000,
                                 "partial_rotary_factor": 0.25,
                                 "mrope_interleaved": True,
                                 "mrope_section": [11, 11, 10]},
                layer_types=["linear_attention"] * 3 + ["full_attention"],
                quant_config=legacy_w8a16_quant_config())
    config.layer_types *= 16

    class Inner:
        def to(self, *args, **kwargs):
            pytest.fail("legacy adapter must not upload the whole text stack")

    class Model:
        def half(self):
            events.append("half")
            return self
        def get_output_embeddings(self):
            return object()

    model = Model()
    model.model, model.config = Inner(), config
    owner = adapter.Qwen3_5Adapter.__new__(adapter.Qwen3_5Adapter)
    owner.model, owner._rpu_is_ready = model, False
    monkeypatch.setattr(adapter, "_claim_live_instance", lambda m: events.append("claim"))
    monkeypatch.setattr(adapter, "_cleanup_failed_text_install", lambda *a: events.append("cleanup"))

    def claim_allocator(enabled):
        assert enabled is False
        events.append("direct-allocator")

    def prepare(policy):
        assert policy.qwen35_legacy_27b_sdk_budget
        events.append("prepare-arenas")
        return True

    def install(inner, passed_config, *args, **kwargs):
        assert inner is model.model and passed_config is config
        events.append("bounded-install")
        raise _StopAtNativeWeights

    monkeypatch.setattr(torch.rpu, "set_caching_allocator", claim_allocator)
    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", prepare)
    monkeypatch.setattr(adapter, "install_qwen3_5_text_for_rpu", install)
    with pytest.raises(_StopAtNativeWeights):
        owner.to_rpu(max_seq_len=128)
    assert events == ["claim", "direct-allocator", "prepare-arenas", "half",
                      "bounded-install", "cleanup"]
    assert model._rpu_swizzle_started


def test_legacy_requires_cpu_sources_before_any_upload(monkeypatch, host_device):
    inner, config = _fixture()
    inner.layers[-1].input_layernorm.weight = torch.nn.Parameter(
        torch.empty(128, device="meta", dtype=torch.float16))
    monkeypatch.setattr(text, "validate_legacy_text_profile", lambda c: True)
    monkeypatch.setattr(torch.nn.Module, "to", lambda *a, **kw: pytest.fail("uploaded invalid sources"))
    with pytest.raises(text.UnsupportedModelError, match="original CPU source weights"):
        _install(inner, config)
