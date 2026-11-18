"""Packed Qwen3 installation must preserve the chosen component topology."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.api.cache import RPUCache
from rpu_backend.quant.qwen3_profiles import qwen3_w4_topology
from rpu_backend.runtime import decoder
from rpu_backend.tests.acceptance import test_decoder_install_transaction as transaction
from test_qwen3_quant_profiles import GEOMETRIES, config
from test_qwen3_w8_core_profiles import _config as w8_config


def _packed_model(geometry, cores):
    from rpu_backend.runtime.topology import decoder_packed_w4_mlp_intermediate_size

    cfg = config(geometry)
    topology = qwen3_w4_topology(cfg, cores)
    h, logical, layers, heads = geometry
    # Config stays logical; the installed MLP6 leaves carry the padded width.
    intermediate = decoder_packed_w4_mlp_intermediate_size(logical, topology.mlp_tp)

    def norm(width):
        return SimpleNamespace(weight=torch.empty(width, device="meta", dtype=torch.float16),
                               variance_epsilon=1e-6)

    def linear(n, k, partition, tp):
        groups, local_n = (k // 32, n // tp) if partition else (k // 32 // tp, n)
        count = tp * ((groups + 3) // 4 * 4) * ((local_n + 63) // 64 * 64)
        return SimpleNamespace(out_features=n, in_features=k,
            weight=torch.empty(n, k // 2, device="meta", dtype=torch.uint8),
            weight_scale=torch.empty(32, count // 32, device="meta", dtype=torch.float16),
            _rpu_linear_partition=partition, _rpu_linear_num_cores=tp)

    attn = SimpleNamespace(head_dim=128, q_norm=norm(128), k_norm=norm(128))
    for name, n, k, partition in (("q_proj", heads * 128, h, 1), ("k_proj", 1024, h, 1),
                                 ("v_proj", 1024, h, 1), ("o_proj", h, heads * 128, 0)):
        setattr(attn, name, linear(n, k, partition, topology.attn_tp))
    mlp = SimpleNamespace(gate_proj=linear(intermediate, h, 1, topology.mlp_tp),
        up_proj=linear(intermediate, h, 1, topology.mlp_tp),
        down_proj=linear(h, intermediate, 0, topology.mlp_tp))
    layer = SimpleNamespace(self_attn=attn, mlp=mlp,
        input_layernorm=norm(h), post_attention_layernorm=norm(h))
    layer.get_submodule = lambda role: getattr(getattr(layer, role.split(".")[0]), role.split(".")[1])
    model = transaction._Model()
    model.config, model.layers, model.norm = cfg, [layer] * layers, norm(h)
    model.forward = lambda **kwargs: None
    scales = [[getattr(container, role).weight_scale] * layers
        for container, role in ((attn, "q_proj"), (attn, "k_proj"), (attn, "v_proj"),
                                (attn, "o_proj"), (mlp, "gate_proj"), (mlp, "up_proj"),
                                (mlp, "down_proj"))]
    return model, topology, scales


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_real_installer_keeps_packed_topology_on_install_replacement_and_cache(monkeypatch, geometry, cores):
    state = transaction.native_install.__wrapped__(monkeypatch)
    model, topology, scales = _packed_model(geometry, cores)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_weights_w8a16",
                        torch.ops.rpu.causal_decoder_set_weights, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_topology",
                        lambda handle: topology.identity()[1:], raising=False)
    table = transaction._FakeTensor()
    options = dict(arch="qwen3", topology=topology, scale_lists=scales,
        cos_sin=(table, table), execution_config={"model": {"num_cores": cores}},
        chunk_envelope_for=lambda *args: (128, 128))
    try:
        for expected_handle in (1, 2):
            handle = decoder._install_causal_decoder_forward(model, **options)
            assert handle == expected_handle
            assert state["alive"] == {handle}
            assert model._rpu_decoder_topology is topology
            assert model._rpu_decoder_graph_cache.runtime_policy.execution_core_count == cores
            assert model.layers[0].mlp.down_proj._rpu_linear_num_cores == topology.mlp_tp
            owner = SimpleNamespace(config=model.config, model=model, _rpu_decoder_topology=topology)
            with torch._subclasses.fake_tensor.FakeTensorMode():
                cache = RPUCache.from_model(owner, max_seq_len=16, device="cpu")
            assert cache.attn_tp == topology.attn_tp
            assert cache.physical_kv_cores == 8
    finally:
        resource = getattr(model, "_rpu_decoder_retirement_state", None)
        if resource is not None:
            resource.retire()
    assert state["alive"] == set()


def _padded_w8_model(geometry):
    from rpu_backend.api._execution import qwen3_core_profile
    from rpu_backend.runtime.topology import decoder_mlp_intermediate_size, resolve_decoder_topology

    model, _, _ = _packed_model(geometry, 6)
    model.config = w8_config(geometry)
    topology = resolve_decoder_topology(num_cores=6, **qwen3_core_profile(model.config, 6).geometry())
    layer = model.layers[0]
    physical = decoder_mlp_intermediate_size(geometry[1], topology.mlp_tp)
    scales = []
    for role in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        projection = layer.get_submodule(role)
        if role in ("mlp.gate_proj", "mlp.up_proj"):
            projection.out_features = physical
        elif role == "mlp.down_proj":
            projection.in_features = physical
        projection._rpu_linear_num_cores = topology.mlp_tp if role.startswith("mlp.") else topology.attn_tp
        projection.weight = torch.empty(projection.out_features, projection.in_features,
                                        dtype=torch.int8, device="meta")
        projection.weight_scale = torch.empty(projection.out_features, dtype=torch.float16, device="meta")
        scales.append([projection.weight_scale] * len(model.layers))
    return model, topology, scales


@pytest.mark.parametrize("geometry", [GEOMETRIES[2], GEOMETRIES[4], GEOMETRIES[5]])
@pytest.mark.parametrize("fault", [None, "scale_identity", "padding_shape"])
def test_real_installer_accepts_only_complete_padded_w8_payload(monkeypatch, geometry, fault):
    state = transaction.native_install.__wrapped__(monkeypatch)
    model, topology, scales = _padded_w8_model(geometry)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_weights_w8a16",
                        torch.ops.rpu.causal_decoder_set_weights, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_topology",
                        lambda handle: topology.identity()[1:], raising=False)
    if fault == "scale_identity":
        scales[-1][-1] = scales[-1][-1].clone()
    elif fault == "padding_shape":
        model.layers[-1].mlp.down_proj.weight = torch.empty(geometry[0], geometry[1],
            dtype=torch.int8, device="meta")
    table = transaction._FakeTensor()
    options = dict(arch="qwen3", topology=topology, scale_lists=scales, cos_sin=(table, table),
        execution_config={"model": {"num_cores": 6}}, chunk_envelope_for=lambda *args: (128, 128))
    if fault:
        with pytest.raises(ValueError, match="actual INT8|physical MLP shape"):
            decoder._install_causal_decoder_forward(model, **options)
        assert state["alive"] == set() and state["events"] == []
    else:
        try:
            assert decoder._install_causal_decoder_forward(model, **options) == 1
            assert model._rpu_decoder_topology == topology
            assert model._rpu_decoder_graph_cache.runtime_policy.execution_core_count == 6
            assert model.layers[-1].mlp.down_proj.weight.dtype == torch.int8
        finally:
            model._rpu_decoder_retirement_state.retire()


@pytest.mark.parametrize("geometry", [GEOMETRIES[2], GEOMETRIES[4]])
@pytest.mark.parametrize("width", ["logical"])
def test_real_installer_rejects_packed_mlp6_without_its_192_aligned_width(monkeypatch, geometry, width):
    """A packed MLP6 payload must carry the shared 32*TP physical width."""
    from rpu_backend.runtime.topology import decoder_mlp_intermediate_size

    state = transaction.native_install.__wrapped__(monkeypatch)
    model, topology, scales = _packed_model(geometry, 6)
    h, logical = geometry[:2]
    stale = logical
    down = model.layers[-1].mlp.down_proj
    groups = stale // 32 // topology.mlp_tp
    count = topology.mlp_tp * ((groups + 3) // 4 * 4) * ((h + 63) // 64 * 64)
    down.in_features = stale
    down.weight = torch.empty(h, stale // 2, device="meta", dtype=torch.uint8)
    down.weight_scale = torch.empty(32, count // 32, device="meta", dtype=torch.float16)
    scales[-1] = [down.weight_scale] * len(model.layers)
    table = transaction._FakeTensor()
    with pytest.raises(ValueError, match="actual UINT8/G32"):
        decoder._install_causal_decoder_forward(model, arch="qwen3", topology=topology,
            scale_lists=scales, cos_sin=(table, table),
            execution_config={"model": {"num_cores": 6}},
            chunk_envelope_for=lambda *args: (128, 128))
    assert state["alive"] == set() and state["events"] == []


@pytest.mark.parametrize("geometry", [GEOMETRIES[2], GEOMETRIES[4], GEOMETRIES[5]])
def test_w4_hot_chunk_reconfigure_retains_padded_mlp6_topology_and_cache_policy(monkeypatch, geometry):
    from rpu_backend.adapters.qwen3 import Qwen3Adapter
    from test_causal_prefill_padding import _ConfigView

    state = transaction.native_install.__wrapped__(monkeypatch)
    model, topology, scales = _packed_model(geometry, 6)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_weights_w8a16",
                        torch.ops.rpu.causal_decoder_set_weights, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_topology",
                        lambda handle: topology.identity()[1:], raising=False)
    table = transaction._FakeTensor()
    execution = {"model": {"num_cores": 6}}
    decoder._install_causal_decoder_forward(model, arch="qwen3", topology=topology,
        scale_lists=scales, cos_sin=(table, table), execution_config=execution,
        chunk_envelope_for=lambda *args: (128, 128))
    owner = _ConfigView(model=model, config=model.config, _rpu_execution=execution,
        _rpu_decoder_topology=topology, _rpu_swizzled=True, _rpu_swizzle_started=True)
    adapter = _ConfigView(model=owner, preflight_execution=Qwen3Adapter.preflight_execution,
                         EXECUTION_SUPPORTED=Qwen3Adapter.EXECUTION_SUPPORTED)
    session = decoder._bind_causal_decoder_execution_session(adapter, entry_point="Qwen3 W4 host install")
    graph_cache = model._rpu_decoder_graph_cache
    cleared = []
    graph_cache.begin_warmup = lambda: None
    graph_cache.clear = lambda: cleared.append(True)
    pending = {}
    def stage(handle, token, chunk):
        assert handle == model._rpu_decoder_handle == 1
        pending[token] = chunk
    def commit(token):
        state["hot_chunk"] = pending.pop(token)
    for name, operation in {
        "execution_reconfigure_begin": lambda token: token,
        "execution_reconfigure_commit": commit,
        "execution_reconfigure_abort": lambda token: pending.pop(token, None),
        "execution_reconfigure_abort_attempt": lambda token: pending.pop(token, None),
        "causal_decoder_stage_chunk_size_override": stage,
    }.items():
        monkeypatch.setattr(torch.ops.rpu, name, operation, raising=False)
    try:
        assert decoder._causal_lm_runtime_complete(owner)
        session.reconfigure({"prefill": {"chunk_size": 32}})
        assert state["hot_chunk"] == 32 and len(cleared) == 1
        assert topology.mlp_tp == 6
        assert model.layers[0].mlp.down_proj.in_features == {
        9728: 9792, 17408: 17472, 25600: 25728}[geometry[1]]
        assert model._rpu_decoder_topology is owner._rpu_decoder_topology is topology
        assert model._rpu_decoder_graph_cache is graph_cache
        assert graph_cache.runtime_policy.execution_core_count == 6
        assert state["alive"] == {1} and len(state["create_options"]) == 1
        assert owner._rpu_execution is model._rpu_execution is session.config
        with pytest.raises(ValueError, match="cold-only"):
            session.reconfigure({"model": {"num_cores": 4}})
    finally:
        model._rpu_decoder_retirement_state.retire()
