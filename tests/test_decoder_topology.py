"""Host regressions for one cold topology across planning, weights and KV."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.api._execution import normalize_rpu_execution, _QWEN3_EXECUTION_SUPPORTED
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime import weights
from rpu_backend.runtime.topology import (
    resolve_decoder_topology, validate_decoder_cache_topology,
)


_FAMILY_ROWS = {
    "0.6b": (1024, 3072, 28, 16),
    "1.7b": (2048, 6144, 28, 16),
    "4b": (2560, 9728, 36, 32),
    "8b": (4096, 12288, 36, 32),
    "14b": (5120, 17408, 40, 40),
    "32b": (5120, 25600, 64, 64),
}
_FP16_ROWS = tuple(size for size in _FAMILY_ROWS if size != "32b")


def _geometry(size="0.6b"):
    hidden, intermediate, _, heads = _FAMILY_ROWS[size]
    return dict(hidden_size=hidden, intermediate_size=intermediate, num_q_heads=heads,
                num_kv_heads=8, head_dim=128, vocab_size=151936)


def _config(size="0.6b", **overrides):
    hidden, intermediate, layers, heads = _FAMILY_ROWS[size]
    values = dict(hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=layers,
                  num_attention_heads=heads, num_key_value_heads=8, head_dim=128,
                  vocab_size=151936, model_type="qwen3")
    values.update(overrides)
    return SimpleNamespace(**values)


def test_six_core_budget_keeps_physical_kv_controllers_and_legal_site_divisors():
    topology = resolve_decoder_topology(num_cores=6, **_geometry())
    assert topology.identity() == (2, 6, 4, 6, 4, 8)
    with pytest.raises(FrozenInstanceError):
        topology.attn_tp = 6
    assert resolve_decoder_topology(num_cores=8, **_geometry()).identity() == (2, 8, 8, 8, 8, 8)


def test_fused_lm_head_reuses_decoder_graph_core_budget(monkeypatch):
    from rpu_backend.adapters import qwen3
    from rpu_backend.runtime.decoder import _new_decoder_graph_cache

    topology = resolve_decoder_topology(num_cores=6, **_geometry())
    graph_cache = _new_decoder_graph_cache(topology)
    inner = SimpleNamespace(_rpu_decoder_handle=17,
        _rpu_decoder_topology=topology, _rpu_decoder_graph_cache=graph_cache)
    causal_lm = SimpleNamespace(model=inner, config=_config())
    pushed = []
    planned = []
    monkeypatch.setattr(qwen3, "_push_lm_head_weight",
                        lambda model, handle: pushed.append((model, handle)))
    monkeypatch.setattr(qwen3, "_text_decode_execution_plan",
                        lambda model, handle: planned.append((model, handle)))
    monkeypatch.setattr(qwen3, "_install_causal_decoder_forward",
                        lambda *_a, **_k: pytest.fail("fused head replaced installed decoder"))

    assert qwen3._apply_fused_lm_head_for_rpu(causal_lm) == 17
    assert pushed == [(causal_lm, 17)]
    assert planned == [(inner, 17)]
    assert causal_lm.model._rpu_decoder_graph_cache is graph_cache
    assert graph_cache.runtime_policy.execution_core_count == 6


def test_model_scope_is_declared_detached_and_immutable():
    value = {"model": {"num_cores": 6}, "prefill": {"chunk_size": 32}}
    config = normalize_rpu_execution(value, entry_point="test", supported=_QWEN3_EXECUTION_SUPPORTED)
    value["model"]["num_cores"] = 8
    assert config["model"]["num_cores"] == 6
    with pytest.raises(TypeError):
        config["model"]["num_cores"] = 8
    with pytest.raises(ValueError, match="unknown stage"):
        normalize_rpu_execution(value, entry_point="unsupported-owner", supported={"prefill": ("chunk_size",)})


@pytest.mark.parametrize("value", [True, 0, 1, 2, 3, 5, 7, 9, "6", 6.0])
def test_invalid_core_request_is_rejected_before_geometry(value):
    with pytest.raises(ValueError, match="model.num_cores"):
        normalize_rpu_execution({"model": {"num_cores": value}},
            entry_point="test", supported=_QWEN3_EXECUTION_SUPPORTED)


@pytest.mark.parametrize("override", [
    {"hidden_size": 2048}, {"num_attention_heads": 8}, {"head_dim": 64},
    {"vocab_size": 151938}, {"quant_config": {"method": "w8a16"}},
    {"quantization_config": {"quant_method": "awq"}},
])
def test_six_core_profile_fails_before_full_weight_load(monkeypatch, override):
    from rpu_backend.api import causal_lm
    from rpu_backend.adapters.qwen3 import Qwen3Adapter
    monkeypatch.setattr(causal_lm.AutoConfig, "from_pretrained", lambda *_a, **_k: _config(**override))
    monkeypatch.setattr(causal_lm, "resolve_config_architecture", lambda *_a, **_k: "Qwen3ForCausalLM")
    monkeypatch.setattr(causal_lm, "get_adapter", lambda _arch: Qwen3Adapter)
    monkeypatch.setattr(causal_lm.AutoModelForCausalLM, "from_pretrained",
                        lambda *_a, **_k: pytest.fail("invalid topology loaded model weights"))
    with pytest.raises(causal_lm.UnsupportedModelError):
        causal_lm.RPUModelForCausalLM.from_pretrained("unused", rpu_execution={"model": {"num_cores": 6}})


def test_weight_walk_binds_each_projection_role_to_its_actual_layout():
    module = torch.nn.Module()
    for name, n, k in (("q_proj", 128, 64), ("o_proj", 64, 128),
                       ("gate_proj", 96, 64), ("up_proj", 96, 64),
                       ("down_proj", 64, 96), ("lm_head", 64, 64)):
        module.add_module(name, torch.nn.Linear(k, n, bias=False).half())
    original = {name: child.weight.detach().clone() for name, child in module.named_children()}
    weights.convert_linear_weights_inplace(module, attn_num_cores=4,
                                           mlp_num_cores=6, lm_head_num_cores=4)
    for name, child in module.named_children():
        cores = 6 if name in {"gate_proj", "up_proj", "down_proj"} else 4
        partition = 0 if name in {"o_proj", "down_proj"} else 1
        assert (child._rpu_linear_partition, child._rpu_linear_num_cores) == (partition, cores)
        assert torch.equal(child.weight, weights.transform_linear_weight(original[name], partition, cores))


def test_four_b_six_core_budget_packs_padded_mlp_with_six_shards():
    topology = resolve_decoder_topology(num_cores=6, **_geometry("4b"))
    module = torch.nn.Module()
    # Keep the real 4B intermediate alignment boundary; a small input width
    # avoids allocating full model weights for this shared packing regression.
    for name, n, k in (("gate_proj", 9792, 64), ("up_proj", 9792, 64),
                       ("down_proj", 64, 9792)):
        module.add_module(name, torch.nn.Linear(k, n, bias=False).half())
    weights.convert_linear_weights_inplace(module, attn_num_cores=topology.attn_tp,
        mlp_num_cores=topology.mlp_tp, lm_head_num_cores=topology.lm_head_tp,
        execution_core_count=topology.num_cores)
    for name, child in module.named_children():
        assert (child._rpu_linear_partition, child._rpu_linear_num_cores) == (
            0 if name == "down_proj" else 1, 6)
    with pytest.raises(ValueError, match="divisible"):
        weights.tp_col_swizzle_mc_weight(torch.zeros(9728, 64, dtype=torch.float16), 6)


def test_nondefault_eager_linear_uses_recorded_core_count(monkeypatch):
    layer = torch.nn.Linear(64, 64, bias=False).half()
    weights._record_linear_layout(layer, 1, 4)
    output = torch.ones(1, 64, dtype=torch.float16)
    calls = []
    monkeypatch.setattr(torch.ops.rpu, "linear_with_core_count",
                        lambda *args: calls.append(args) or output, raising=False)
    value = SimpleNamespace(device=SimpleNamespace(type="rpu"), dtype=torch.float16,
                            dim=lambda: 2, size=lambda _d: 64, contiguous=lambda: output)
    assert layer(value) is output
    assert calls[0][1] is layer.weight and calls[0][-2:] == (1, 4)


@pytest.mark.parametrize("installed", [False, True])
def test_cache_factory_consumes_attention_tp_and_keeps_eight_controller_layout(installed):
    model = SimpleNamespace(config=_config(),
                            _rpu_execution={"model": {"num_cores": 6}})
    topology = resolve_decoder_topology(num_cores=6, **_geometry())
    if installed:
        model._rpu_decoder_topology = topology
    cache = RPUCache.from_model(model, max_seq_len=16, device="cpu")
    assert (cache.attn_tp, cache.physical_kv_cores) == (4, 8)
    assert cache.nKVHeadVx * cache.nKVHeadChunk == 16
    validate_decoder_cache_topology(topology, cache)
    stale = RPUCache(1, 1, 16, 8, 128, device="cpu")
    with pytest.raises(ValueError, match="topology does not match"):
        validate_decoder_cache_topology(topology, stale)


def test_reduced_core_batch_is_rejected_before_allocation_and_forward(monkeypatch):
    model = SimpleNamespace(config=_config(), _rpu_execution={"model": {"num_cores": 6}})
    topology = resolve_decoder_topology(num_cores=6, **_geometry())
    direct = RPUCache(1, 2, 16, 8, 128, attn_tp=4, device="cpu")
    monkeypatch.setattr(torch, "zeros", lambda *_a, **_k: pytest.fail("unsupported batch allocated KV buffers"))
    with pytest.raises(ValueError, match="batch_size=1"):
        RPUCache.from_model(model, batch_size=2, max_seq_len=16, device="cpu")
    with pytest.raises(ValueError, match="batch_size=1"):
        validate_decoder_cache_topology(topology, direct)


@pytest.mark.parametrize("size", _FAMILY_ROWS)
@pytest.mark.parametrize("cores, expected", [(4, (2, 4, 4, 4, 4, 8)),
    (6, (2, 6, 4, 6, 4, 8)), (8, (2, 8, 8, 8, 8, 8))])
def test_fixed_templates_are_shared_by_real_adapter_preflight_and_cache(cores, expected, size):
    from rpu_backend.adapters.qwen3 import _execution_topology, Qwen3Adapter
    config = _config(size)
    execution = {"model": {"num_cores": cores}}
    Qwen3Adapter.preflight_execution(config, execution)
    topology = _execution_topology(config, execution)
    assert topology.identity() == expected
    model = SimpleNamespace(config=config, _rpu_execution=execution,
                            _rpu_decoder_topology=topology)
    cache = RPUCache.from_model(model, max_seq_len=16, device="cpu")
    validate_decoder_cache_topology(topology, cache)
    assert cache.attn_tp == expected[2]
    assert cache.physical_kv_cores == 8 and cache.num_layers == config.num_hidden_layers
    if cores != 8:
        with pytest.raises(ValueError, match="batch_size=1"):
            RPUCache.from_model(model, max_seq_len=16, device="cpu", batch_size=2)


def test_unknown_linear_late_in_tree_rejects_before_any_swizzle(monkeypatch):
    model = torch.nn.Module()
    model.add_module("q_proj", torch.nn.Linear(64, 64, bias=False).half())
    model.add_module("unknown", torch.nn.Linear(64, 64, bias=False).half())
    original = model.q_proj.weight.detach().clone()
    monkeypatch.setattr(weights, "transform_linear_weight",
                        lambda *_a, **_k: pytest.fail("swizzled before complete role admission"))
    with pytest.raises(ValueError, match="unknown Linear role"):
        weights.convert_linear_weights_inplace(model, attn_num_cores=4,
            mlp_num_cores=6, lm_head_num_cores=4, execution_core_count=6)
    assert torch.equal(model.q_proj.weight, original)
    assert not hasattr(model.q_proj, "_rpu_linear_partition")


@pytest.mark.parametrize("override, message", [
    ({"model_type": "llama"}, "exact Llama-3.2-1B"),
    ({"model_type": "mistral"}, "plain FP16 Qwen3"), ({"num_hidden_layers": 27}, "plain FP16 Qwen3"),
    ({"quant_config": {"method": "w8a16"}}, "plain FP16 Qwen3"),
    ({"rope_parameters": {"rope_type": "linear"}}, "plain FP16 Qwen3")])
def test_uninstalled_cache_rejects_unsupported_profile_before_allocation(monkeypatch, override, message):
    model = SimpleNamespace(config=_config(**override), _rpu_execution={"model": {"num_cores": 4}})
    monkeypatch.setattr(torch, "zeros", lambda *_a, **_k: pytest.fail("unsupported profile allocated KV"))
    with pytest.raises(ValueError, match=message):
        RPUCache.from_model(model, max_seq_len=16, device="cpu")


@pytest.mark.parametrize("size", _FP16_ROWS)
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_public_loader_keeps_cold_request_through_real_qwen_adapter(monkeypatch, cores, size):
    from rpu_backend.api import causal_lm
    from rpu_backend.adapters.qwen3 import Qwen3Adapter
    from transformers import Qwen3Config, Qwen3ForCausalLM
    with torch.device("meta"):
        model = Qwen3ForCausalLM(Qwen3Config(**vars(_config(size)))).half()
    calls = []
    monkeypatch.setattr(causal_lm.AutoConfig, "from_pretrained", lambda *_a, **_k: model.config)
    monkeypatch.setattr(causal_lm, "resolve_config_architecture", lambda *_a, **_k: "Qwen3ForCausalLM")
    monkeypatch.setattr(causal_lm, "get_adapter", lambda _arch: Qwen3Adapter)
    monkeypatch.setattr(causal_lm.AutoModelForCausalLM, "from_pretrained",
                        lambda *_a, **_k: calls.append("load") or model)
    returned = causal_lm.RPUModelForCausalLM.from_pretrained(
        "unused", rpu_execution={"model": {"num_cores": cores}})
    assert returned is model and calls == ["load"]
    assert model._rpu_execution["model"] == {"num_cores": cores}
    assert model.model._rpu_execution is model._rpu_execution
    assert not getattr(model, "_rpu_swizzle_started", False)


@pytest.mark.parametrize("cores", [4, 6])
def test_reduced_batch_rejects_before_embedding_allocation(cores):
    from rpu_backend.runtime.decoder import _run_causal_decoder_forward
    topology = resolve_decoder_topology(num_cores=cores, **_geometry())
    model = SimpleNamespace(_rpu_decoder_topology=topology,
        embed_tokens=lambda *_: pytest.fail("unsupported batch allocated embedding"))
    cache = RPUCache(1, 1, 16, 8, 128, attn_tp=4, device="cpu")
    with pytest.raises(ValueError, match="batch_size=1"):
        _run_causal_decoder_forward(model, 7, input_ids=torch.zeros(2, 1, dtype=torch.long),
                                   past_key_values=cache)


def test_private_topology_rejects_unadmitted_geometry_and_arbitrary_role_tuple():
    from rpu_backend.runtime.topology import DecoderTopology
    with pytest.raises(ValueError, match="reduced-core decoder template"):
        resolve_decoder_topology(num_cores=6, **(_geometry() | {"hidden_size": 2048}))
    with pytest.raises(ValueError, match="fixed core template"):
        DecoderTopology(6, 4, 5, 4)


def test_cache_factory_reads_installed_inner_topology_and_rejects_conflicting_outer():
    topology = resolve_decoder_topology(num_cores=6, **_geometry())
    inner = SimpleNamespace(_rpu_decoder_handle=7, _rpu_decoder_topology=topology)
    model = SimpleNamespace(config=_config(), model=inner)
    cache = RPUCache.from_model(model, max_seq_len=16, device="cpu")
    assert cache.attn_tp == 4
    model._rpu_decoder_topology = resolve_decoder_topology(num_cores=4, **_geometry())
    with pytest.raises(ValueError, match="topology differ"):
        RPUCache.from_model(model, max_seq_len=16, device="cpu")


@pytest.mark.parametrize("size", _FAMILY_ROWS)
def test_fused_head_uses_installed_inner_topology_before_transform(monkeypatch, size):
    from rpu_backend.adapters import qwen3
    topology = resolve_decoder_topology(num_cores=6, **_geometry(size))
    model = SimpleNamespace(model=SimpleNamespace(_rpu_decoder_handle=7,
        _rpu_decoder_topology=topology), config=_config(size),
        lm_head=torch.nn.Linear(_config(size).hidden_size, 151936, bias=False, device="meta", dtype=torch.float16))
    calls = []
    sentinel = object()
    def transform(weight, *, partition, num_cores):
        calls.append((partition, num_cores))
        return SimpleNamespace(to=lambda device: sentinel)
    monkeypatch.setattr(qwen3, "transform_linear_weight", transform)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_lm_head",
                        lambda handle, weight: calls.append((handle, weight)), raising=False)
    qwen3._push_lm_head_weight(model, 7)
    assert calls == [(1, 4), (7, sentinel)]


@pytest.mark.parametrize("scope", [{}, {"num_cores": 8}, {"num_cores": 4}])
def test_public_loader_rejects_undeclared_owner_model_scope_before_weight_load(monkeypatch, scope):
    from rpu_backend.api import causal_lm

    class PrefillOnlyAdapter:
        """A CausalLM owner that declares no EXECUTION_SUPPORTED model scope."""
        preflight = classmethod(lambda cls, config: pytest.fail("undeclared scope reached preflight"))

    monkeypatch.setattr(causal_lm.AutoConfig, "from_pretrained", lambda *_a, **_k: _config(model_type="llama"))
    monkeypatch.setattr(causal_lm, "resolve_config_architecture", lambda *_a, **_k: "LlamaForCausalLM")
    monkeypatch.setattr(causal_lm, "get_adapter", lambda _arch: PrefillOnlyAdapter)
    monkeypatch.setattr(causal_lm.AutoModelForCausalLM, "from_pretrained",
                        lambda *_a, **_k: pytest.fail("unsupported owner loaded model weights"))
    with pytest.raises(ValueError, match="unknown stage"):
        causal_lm.RPUModelForCausalLM.from_pretrained("unused", rpu_execution={"model": scope})


@pytest.mark.parametrize("cores", [4, 6])
def test_llama_owner_rejects_non_llama_geometry_before_weight_load(monkeypatch, cores):
    # Llama declares the model scope; a Qwen-shaped "llama" config still fails
    # its exact profile before any checkpoint weight is loaded.
    from rpu_backend.api import causal_lm
    from rpu_backend.adapters.llama import LlamaAdapter
    monkeypatch.setattr(causal_lm.AutoConfig, "from_pretrained", lambda *_a, **_k: _config(model_type="llama"))
    monkeypatch.setattr(causal_lm, "resolve_config_architecture", lambda *_a, **_k: "LlamaForCausalLM")
    monkeypatch.setattr(causal_lm, "get_adapter", lambda _arch: LlamaAdapter)
    monkeypatch.setattr(causal_lm.AutoModelForCausalLM, "from_pretrained",
                        lambda *_a, **_k: pytest.fail("unsupported owner loaded model weights"))
    with pytest.raises(causal_lm.UnsupportedModelError):
        causal_lm.RPUModelForCausalLM.from_pretrained("unused", rpu_execution={"model": {"num_cores": cores}})


@pytest.mark.parametrize("size", _FP16_ROWS)
@pytest.mark.parametrize("cores", [4, 6])
@pytest.mark.parametrize("phase", ["constructor", "before_to_rpu"])
@pytest.mark.parametrize("fault", ["layer_count", "head", "last_projection", "embedding", "norm", "extra_linear", "dtype", "bias"])
def test_actual_reduced_model_is_rejected_before_claim_or_swizzle(monkeypatch, cores, phase, fault, size):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from rpu_backend.adapters import qwen3
    with torch.device("meta"):
        model = Qwen3ForCausalLM(Qwen3Config(**vars(_config(size)))).half()
    hidden, intermediate = model.config.hidden_size, model.config.intermediate_size
    model._rpu_execution = {"model": {"num_cores": cores}}
    adapter = qwen3.Qwen3Adapter(model) if phase == "before_to_rpu" else None
    with torch.device("meta"):
        if fault == "layer_count":
            model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:-1])
        elif fault == "head":
            model.lm_head = torch.nn.Linear(hidden, 151808, bias=False, dtype=torch.float16)
        elif fault == "last_projection":
            model.model.layers[-1].mlp.down_proj = torch.nn.Linear(intermediate, hidden - 64, bias=False, dtype=torch.float16)
        elif fault == "embedding":
            model.model.embed_tokens = torch.nn.Embedding(151808, hidden, dtype=torch.float16)
        elif fault == "norm":
            model.model.layers[-1].post_attention_layernorm.weight = torch.nn.Parameter(torch.empty(hidden - 64, dtype=torch.float16))
        elif fault == "extra_linear":
            model.extra = torch.nn.Linear(hidden, hidden, bias=False, dtype=torch.float16)
        elif fault == "dtype":
            model.model.layers[-1].self_attn.k_norm.weight = torch.nn.Parameter(torch.empty(128, dtype=torch.float32))
        else:
            model.model.layers[-1].mlp.down_proj.bias = torch.nn.Parameter(torch.zeros(hidden, dtype=torch.float16))
    monkeypatch.setattr(qwen3, "_claim_live_instance", lambda *_: pytest.fail("invalid structure claimed device model"))
    monkeypatch.setattr(qwen3, "convert_linear_weights_inplace", lambda *_a, **_k: pytest.fail("invalid structure started conversion"))
    with pytest.raises(qwen3.UnsupportedModelError, match="decoder|FP16 model weights"):
        if adapter is None:
            qwen3.Qwen3Adapter(model)
        else:
            adapter.to_rpu()
    assert not getattr(model, "_rpu_swizzle_started", False)
    assert not hasattr(model.model, "_rpu_decoder_handle")


def test_reduced_config_rejects_explicit_sliding_layer_types():
    from transformers import Qwen3Config
    from rpu_backend.adapters.qwen3 import _execution_topology, UnsupportedModelError
    config = Qwen3Config(**vars(_config()), layer_types=["sliding_attention"] * 28,
                         use_sliding_window=False, sliding_window=64)
    with pytest.raises(UnsupportedModelError, match="plain FP16"):
        _execution_topology(config, {"model": {"num_cores": 6}})


@pytest.mark.parametrize("replacement_size", ["4b", "8b"])
def test_six_core_family_change_rejects_before_claim_or_swizzle(monkeypatch, replacement_size):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from rpu_backend.adapters import qwen3
    with torch.device("meta"):
        model = Qwen3ForCausalLM(Qwen3Config(**vars(_config("1.7b")))).half()
        replacement = Qwen3ForCausalLM(Qwen3Config(**vars(_config(replacement_size)))).half()
    model._rpu_execution = {"model": {"num_cores": 6}}
    adapter = qwen3.Qwen3Adapter(model)
    assert adapter._topology.identity() == (2, 6, 4, 6, 4, 8)
    # The role tuple can now match across models. Logical geometry must still
    # remain bound, including the newly padded 4B path.
    model.config, model.model, model.lm_head = replacement.config, replacement.model, replacement.lm_head
    monkeypatch.setattr(qwen3, "_claim_live_instance", lambda *_: pytest.fail("changed topology claimed device"))
    monkeypatch.setattr(qwen3, "convert_linear_weights_inplace", lambda *_a, **_k: pytest.fail("changed topology swizzled weights"))
    with pytest.raises(qwen3.UnsupportedModelError, match="bound cold topology"):
        adapter.to_rpu()
    assert not getattr(model, "_rpu_swizzle_started", False)


@pytest.mark.parametrize("size", ["1.7b", "4b", "8b"])
@pytest.mark.parametrize("fault", [None, "shape", "dtype", "layout"])
def test_larger_fused_head_checks_actual_rpu_weight_before_native_publication(monkeypatch, size, fault):
    from rpu_backend.adapters import qwen3
    topology = resolve_decoder_topology(num_cores=6, **_geometry(size))
    config = _config(size)
    weight = SimpleNamespace(shape=(151936, config.hidden_size), dtype=torch.float16,
        device=SimpleNamespace(type="rpu"))
    weight.detach = lambda: weight
    weight.to = lambda *_: weight
    weight.contiguous = lambda: weight
    head = SimpleNamespace(weight=weight, _rpu_linear_num_cores=4, _rpu_linear_partition=1)
    model = SimpleNamespace(config=config, lm_head=head,
        model=SimpleNamespace(_rpu_decoder_handle=7, _rpu_decoder_topology=topology))
    if fault == "shape": weight.shape = (151808, config.hidden_size)
    elif fault == "dtype": weight.dtype = torch.float32
    elif fault == "layout": head._rpu_linear_num_cores = 8
    calls = []
    monkeypatch.setattr(qwen3, "transform_linear_weight", lambda *_a, **_k: pytest.fail("RPU head was swizzled twice"))
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_lm_head", lambda *args: calls.append(args), raising=False)
    if fault:
        with pytest.raises(qwen3.RPUBackendError, match="FP16|layout differs"):
            qwen3._push_lm_head_weight(model, 7)
        assert calls == []
    else:
        qwen3._push_lm_head_weight(model, 7)
        assert calls == [(7, weight)]
