"""Reduced VL text ownership, admission and rollback without a device."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_vl.text import install_qwen3_vl_text_for_rpu
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime import decoder
from rpu_backend.runtime.topology import decoder_mlp_intermediate_size, resolve_decoder_topology
from rpu_backend.tests.acceptance import test_decoder_install_transaction as dec


_SIZES = {"2b": (2048, 6144, 28, 16), "4b": (2560, 9728, 36, 32),
          "8b": (4096, 12288, 36, 32)}


def _model(size="2b", cores=6):
    hidden, intermediate, layers, heads = _SIZES[size]
    config = SimpleNamespace(model_type="qwen3_vl_text", hidden_size=hidden,
        intermediate_size=intermediate, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=8, head_dim=128,
        vocab_size=151936, hidden_act="silu", rms_norm_eps=1e-6,
        attention_bias=False, tie_word_embeddings=size != "8b", use_cache=True,
        rope_theta=5_000_000.0, max_position_embeddings=262144,
        rope_parameters={"rope_type": "default", "rope_theta": 5_000_000.0,
                         "mrope_interleaved": True, "mrope_section": [24, 20, 20]})
    topology = resolve_decoder_topology(num_cores=cores, hidden_size=hidden,
        intermediate_size=intermediate, num_q_heads=heads, num_kv_heads=8,
        head_dim=128, vocab_size=151936)
    physical = decoder_mlp_intermediate_size(intermediate, topology.mlp_tp)

    def norm(width):
        return SimpleNamespace(weight=torch.empty(width, device="meta", dtype=torch.float16),
                               variance_epsilon=1e-6)

    def linear(name, outputs, inputs, tp):
        return SimpleNamespace(weight=torch.empty((outputs, inputs), device="meta", dtype=torch.float16),
            out_features=outputs, in_features=inputs, bias=None,
            _rpu_linear_num_cores=tp, _rpu_linear_partition=0 if name in {"o_proj", "down_proj"} else 1)

    model = dec._Model()
    model.config = config
    model.norm = norm(hidden)
    model.rotary_emb = SimpleNamespace(config=config, rope_type="default",
        attention_scaling=1.0, mrope_section=[24, 20, 20])
    model.forward = lambda *a, **kw: "original"
    model.layers = []
    for _ in range(layers):
        attn = SimpleNamespace(head_dim=128, q_norm=norm(128), k_norm=norm(128))
        for name, n, k in (("q_proj", heads * 128, hidden), ("k_proj", 1024, hidden),
                           ("v_proj", 1024, hidden), ("o_proj", hidden, heads * 128)):
            setattr(attn, name, linear(name, n, k, topology.attn_tp))
        mlp = SimpleNamespace(act_fn=torch.nn.SiLU())
        for name, n, k in (("gate_proj", physical, hidden), ("up_proj", physical, hidden),
                           ("down_proj", hidden, physical)):
            setattr(mlp, name, linear(name, n, k, topology.mlp_tp))
        model.layers.append(SimpleNamespace(self_attn=attn, mlp=mlp,
            input_layernorm=norm(hidden), post_attention_layernorm=norm(hidden)))
    return model, topology


@pytest.fixture
def native(monkeypatch):
    state = dec.native_install.__wrapped__(monkeypatch)
    def tables(config, **kwargs):
        table = dec._FakeTensor(kwargs.get("max_seq_len") or config.max_position_embeddings)
        return table, table
    monkeypatch.setattr(decoder, "build_mrope_cos_sin_tables", tables)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_topology",
        lambda _handle: (state["create_topologies"][-1]["num_cores"], 4,
                        state["create_topologies"][-1]["num_cores"], 4, 8), raising=False)
    return state


def _install(model, bound_topology, **overrides):
    options = dict(execution_config={"model": {"num_cores": bound_topology.num_cores}},
                   topology=bound_topology, deepstack_lang_layers=[0, 1, 2], max_seq_len=32)
    options.update(overrides)
    return install_qwen3_vl_text_for_rpu(model, **options)


@pytest.mark.parametrize("size", _SIZES)
@pytest.mark.parametrize("cores", (4, 6))
def test_installed_text_cache_native_and_weight_layout_share_the_cold_budget(native, size, cores):
    model, topology = _model(size, cores)
    assert _install(model, topology) == 1
    assert model._rpu_decoder_topology is topology
    assert model._rpu_deepstack_lang_layers == [0, 1, 2]
    assert model._rpu_decoder_retirement_state.keepalive[-2:] == [[24, 20, 20], [0, 1, 2]]
    assert model._rpu_batch_decode_enabled is False
    assert native["create_topologies"] == [{"num_cores": cores, "vocab_size": 151936}]
    assert native["chunk_overrides"] == [(1, 0)]
    assert model._rpu_decoder_graph_cache.runtime_policy.execution_core_count == cores
    assert model._rpu_text_graph_cache.runtime_policy.execution_core_count == cores
    with torch._subclasses.fake_tensor.FakeTensorMode():
        cache = RPUCache.from_model(model, max_seq_len=16, device="cpu")
    assert (cache.attn_tp, cache.physical_kv_cores) == (4, 8)
    assert cache.nKVHeadVx * cache.nKVHeadChunk == 16
    assert model.config.intermediate_size == _SIZES[size][1]
    assert model.layers[-1].mlp.up_proj.out_features == (
        9792 if size == "4b" and cores == 6 else _SIZES[size][1])
    model._rpu_decoder_retirement_state.retire()
    assert native["alive"] == set()


@pytest.mark.parametrize("fault", (
    "arch", "layers", "rope", "deepstack", "fractional_deepstack", "disabled_deepstack",
    "w8", "budget", "missing_topology", "late_dtype", "late_shape", "late_norm", "override_config",
    "late_epsilon", "late_activation", "live_rope",
))
def test_reduced_admission_rejects_before_rope_allocation_or_publication(native, monkeypatch, fault):
    model, topology = _model()
    options = {}
    if fault == "arch":
        model.config.model_type = "qwen3"
    elif fault == "layers":
        model.layers.pop()
    elif fault == "rope":
        model.config.rope_parameters = {**model.config.rope_parameters, "mrope_section": [32, 16, 16]}
    elif fault == "deepstack":
        options["deepstack_lang_layers"] = [5, 11, 17]
    elif fault == "fractional_deepstack":
        options["deepstack_lang_layers"] = [0.1, 1, 2]
    elif fault == "disabled_deepstack":
        options["enable_deepstack"] = False
    elif fault == "w8":
        options["scale_lists"] = ()
    elif fault == "budget":
        options["execution_config"] = {"model": {"num_cores": 4}}
    elif fault == "missing_topology":
        options["topology"] = None
    elif fault == "late_dtype":
        model.layers[-1].mlp.down_proj.weight = model.layers[-1].mlp.down_proj.weight.float()
    elif fault == "late_shape":
        model.layers[-1].self_attn.v_proj.in_features = 1
    elif fault == "late_norm":
        model.layers[-1].post_attention_layernorm.weight = torch.empty(1, device="meta", dtype=torch.float16)
    elif fault == "override_config":
        options["text_config"] = SimpleNamespace(**{**vars(model.config), "rope_theta": 1.0,
            "rope_parameters": {**model.config.rope_parameters, "rope_theta": 1.0}})
    elif fault == "late_epsilon":
        model.layers[-1].self_attn.k_norm.variance_epsilon = 1e-5
    elif fault == "late_activation":
        model.layers[-1].mlp.act_fn = torch.nn.ReLU()
    elif fault == "live_rope":
        model.rotary_emb.mrope_section = [32, 16, 16]
    before = dict(vars(model))
    monkeypatch.setattr(decoder, "build_mrope_cos_sin_tables",
                        lambda *a, **kw: pytest.fail("invalid profile allocated RoPE tables"))
    with pytest.raises(ValueError):
        _install(model, topology, **options)
    assert vars(model) == before and native["events"] == []


def test_direct_shared_installer_cannot_bypass_deepstack_admission(native):
    model, topology = _model()
    table = dec._FakeTensor()
    with pytest.raises(ValueError, match="DeepStack consumers"):
        decoder._install_causal_decoder_forward(model, arch="qwen3_vl_text",
            topology=topology, execution_config={"model": {"num_cores": 6}},
            mrope_section=[24, 20, 20], cos_sin=(table, table),
            deepstack_lang_layers=[5, 11, 17], chunk_envelope_for=lambda *a: (336, 320))
    assert native["events"] == []


def test_native_topology_mismatch_retires_pending_handle_and_restores_old_graphs(native, monkeypatch):
    model, topology = _model()
    _install(model, topology)
    before = dict(vars(model))
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_topology", lambda handle: (6, 4, 4, 4, 8))
    with pytest.raises(RuntimeError, match="native decoder topology differs"):
        _install(model, topology)
    assert vars(model) == before
    assert native["alive"] == {1} and native["events"][-1] == ("destroy", 2)
    model._rpu_decoder_retirement_state.retire()


def test_replacement_rejects_changed_budget_before_allocating_rope(native, monkeypatch):
    model, topology = _model()
    _install(model, topology)
    before = dict(vars(model))
    four = resolve_decoder_topology(num_cores=4, hidden_size=2048, intermediate_size=6144,
        num_q_heads=16, num_kv_heads=8, head_dim=128, vocab_size=151936)
    monkeypatch.setattr(decoder, "build_mrope_cos_sin_tables",
                        lambda *a, **kw: pytest.fail("changed topology allocated RoPE tables"))
    with pytest.raises(ValueError, match="replacement cannot change"):
        _install(model, four)
    assert vars(model) == before and native["alive"] == {1}
    model._rpu_decoder_retirement_state.retire()


@pytest.mark.parametrize("fault", ("cache_tp", "cache_batch", "input_batch"))
def test_installed_forward_rejects_stale_cache_and_batch_before_transfer_or_capture(native, monkeypatch, fault):
    model, topology = _model()
    _install(model, topology)
    cache = RPUCache(1, 2 if fault == "cache_batch" else 1, 16, 8, 128,
                     attn_tp=8 if fault == "cache_tp" else 4, device="cpu")
    ids = torch.zeros((2 if fault == "input_batch" else 1, 1), dtype=torch.int64)
    mask = torch.ones_like(ids)
    monkeypatch.setattr(decoder, "_run_causal_decoder_forward",
                        lambda *a, **kw: pytest.fail("invalid cache reached decoder"))
    monkeypatch.setattr(model._rpu_decoder_graph_cache, "capture",
                        lambda *a, **kw: pytest.fail("invalid cache captured Graph"), raising=False)
    with pytest.raises(ValueError, match="topology does not match|batch_size=1"):
        model.forward(input_ids=ids, attention_mask=mask, past_key_values=cache)
    assert native["alive"] == {1}
    model._rpu_decoder_retirement_state.retire()


@pytest.mark.parametrize("dtype", (torch.float16, torch.int8))
@pytest.mark.parametrize("kv_only", (False, True))
def test_actual_cold_text_installer_binds_lingbot_kv_only_envelope(
        native, monkeypatch, dtype, kv_only):
    """Exercise both installer layers through the actual native cold binding."""
    model, _ = _model("4b", 8)
    scales = None
    if dtype == torch.int8:
        scales = [[] for _ in range(7)]
        for layer in model.layers:
            leaves = [layer.self_attn.q_proj, layer.self_attn.k_proj,
                layer.self_attn.v_proj, layer.self_attn.o_proj,
                layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj]
            for role, leaf in enumerate(leaves):
                leaf.weight = leaf.weight.to(dtype=torch.int8)
                scales[role].append(torch.empty(leaf.out_features, device="meta", dtype=torch.float16))
    model._lingbot_runtime_norm_folded = kv_only
    # The existing native double implements FP16 weights. Reuse the same
    # handle/event semantics for the actual W8 setter selected by this caller.
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_weights_w8a16",
        torch.ops.rpu.causal_decoder_set_weights, raising=False)
    envelopes, modes = [], []
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_chunk_envelope",
        lambda handle, max_kv, chunk: envelopes.append((handle, max_kv, chunk)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_lingbot_prefill_mode",
        lambda handle, folded, cache_only: modes.append((handle, folded, cache_only)), raising=False)
    assert install_qwen3_vl_text_for_rpu(model,
        max_seq_len=512, deepstack_lang_layers=[0, 1, 2], scale_lists=scales,
        folded_rmsnorm=kv_only, prefill_kv_only=kv_only) == 1
    assert envelopes == [(1, 512, 400 if kv_only else 256)]
    assert modes == ([(1, True, True)] if kv_only else [])
    assert bool(model._rpu_prefill_kv_only) is kv_only
    assert native["chunk_overrides"] == [(1, 0)]  # AUTO, never an unsafe override
    model._rpu_decoder_retirement_state.retire()
    assert native["alive"] == set()


@pytest.mark.parametrize("fault", ("hidden", "heads", "cores"))
def test_lingbot_kv_only_envelope_rejects_wrong_cold_owner_before_install(
        native, monkeypatch, fault):
    model, _ = _model("4b", 8)
    model._lingbot_runtime_norm_folded = True
    if fault == "hidden":
        model.config.hidden_size = 2048
    if fault == "heads":
        model.config.num_attention_heads = 16
    options = {"execution_config": {"model": {"num_cores": 4}}} if fault == "cores" else {}
    before = dict(vars(model))
    monkeypatch.setattr(decoder, "build_mrope_cos_sin_tables",
                        lambda *a, **kw: pytest.fail("wrong KV-only owner allocated RoPE"))
    with pytest.raises(ValueError, match="exact ordinary 4B TP8"):
        install_qwen3_vl_text_for_rpu(model, folded_rmsnorm=True,
            prefill_kv_only=True, deepstack_lang_layers=[0, 1, 2], **options)
    assert vars(model) == before and native["events"] == []
