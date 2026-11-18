"""Host regressions for exact FP16 Llama-3.2-1B on the shared 4/6/8-core template.

These check admission, the adapter-to-installer wiring, the one-time MLP6
padding and the cache/attention split on CPU. They are not RPU kernel or board
evidence; the 4/6-core path still needs a saved board run.
"""
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from rpu_backend.api._execution import (
    _LLAMA_EXECUTION_SUPPORTED, llama_core_profile, normalize_rpu_execution,
    qwen3_core_profile, qwen3_vl_text_core_profile,
)
from rpu_backend.api.cache import RPUCache
from rpu_backend.api.errors import UnsupportedModelError
from rpu_backend.adapters import llama
from rpu_backend.runtime import weights
from rpu_backend.runtime.topology import (
    decoder_mlp_intermediate_size, resolve_decoder_topology, validate_decoder_cache_topology,
)

_TEMPLATES = {4: (2, 4, 4, 4, 4, 8), 6: (2, 6, 4, 6, 4, 8)}


def _config(**overrides):
    # Llama-3.2-1B config.json (unsloth/Llama-3.2-1B, the registered asset).
    values = dict(
        hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
        num_attention_heads=32, num_key_value_heads=8, head_dim=64,
        vocab_size=128256, rms_norm_eps=1e-5, tie_word_embeddings=True,
        hidden_act="silu", attention_bias=False, mlp_bias=False,
        max_position_embeddings=131072, rope_theta=500000.0,
        rope_scaling={"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                      "original_max_position_embeddings": 8192, "rope_type": "llama3"})
    values.update(overrides)
    return LlamaConfig(**values)


def _execution(cores):
    return normalize_rpu_execution({"model": {"num_cores": cores}}, entry_point="test",
                                   supported=_LLAMA_EXECUTION_SUPPORTED)


def _meta_model(cores, **overrides):
    with torch.device("meta"):
        model = LlamaForCausalLM(_config(**overrides)).half()
    model._rpu_execution = _execution(cores)
    return model


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_exact_profile_binds_the_shared_template_and_eight_keeps_no_topology(cores):
    config = _config()
    llama.LlamaAdapter.preflight_execution(config, _execution(cores))
    topology = llama._execution_topology(config, _execution(cores))
    if cores == 8:
        # Default eight-core install passes no reduced topology at all.
        assert topology is None
        assert llama._execution_topology(config, {}) is None
        return
    assert topology.identity() == _TEMPLATES[cores]
    # Attention TP4: 8 Q + 2 KV heads per core, as in the Qwen3 4/6 rows.
    assert config.num_attention_heads // topology.attn_tp == 8
    assert config.num_key_value_heads // topology.attn_tp == 2
    # Eager vocabulary head on four cores: 32064 rows per core.
    assert config.vocab_size % (16 * topology.lm_head_tp) == 0
    assert config.vocab_size // topology.lm_head_tp == 32064
    physical = decoder_mlp_intermediate_size(config.intermediate_size, topology.mlp_tp)
    assert physical == (8256 if cores == 6 else 8192)
    assert physical % (16 * topology.mlp_tp) == 0


@pytest.mark.parametrize("cores", [4, 6])
@pytest.mark.parametrize("override", [
    {"attention_bias": True}, {"mlp_bias": True}, {"hidden_act": "gelu"},
    {"rope_scaling": {"rope_type": "dynamic", "factor": 2.0}},
    {"head_dim": 128, "num_attention_heads": 16},
    {"vocab_size": 128000}, {"intermediate_size": 8256},
])
def test_reduced_budget_rejects_inexact_llama_semantics_before_loading(cores, override):
    config = _config(**override)
    with pytest.raises(UnsupportedModelError):
        llama.LlamaAdapter.preflight_execution(config, _execution(cores))
    # Eight cores keep the existing admission, which does not read these fields.
    if "intermediate_size" not in override and "num_attention_heads" not in override:
        llama.LlamaAdapter.preflight_execution(config, _execution(8))


def test_reduced_budget_rejects_quantized_or_wrong_model_type_config():
    for config in (SimpleNamespace(**{**_config().to_dict(), "model_type": "qwen3"}),
                   SimpleNamespace(**{**_config().to_dict(), "quantization_config": {"quant_method": "awq"}})):
        with pytest.raises(ValueError, match="exact Llama-3.2-1B"):
            llama_core_profile(config, 4)
        assert llama_core_profile(config, 8) is None
    assert llama_core_profile(_config(rope_scaling=None), 6).qk_norm is False


def test_qwen_admission_does_not_inherit_the_llama_geometry_row():
    from rpu_backend.adapters import qwen3
    assert (2048, 8192, 16, 8) not in qwen3._SUPPORTED_PROFILES
    fields = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "vocab_size")
    geometry = {key: getattr(_config(), key) for key in fields}
    with pytest.raises(ValueError, match="plain FP16 Qwen3"):
        qwen3_core_profile(SimpleNamespace(**geometry, model_type="qwen3"), 4)
    with pytest.raises(ValueError, match="Qwen3-VL"):
        qwen3_vl_text_core_profile(SimpleNamespace(**geometry, model_type="qwen3_vl_text"), 4)


def _shard(physical, partition, cores, core):
    """Read one core's stripe directly from the swizzled buffer, as the kernel does."""
    n, k = physical.shape
    if partition == 1:   # col: [N/16/C, K/16, C, 16, 16]
        block = physical.view(n // 16 // cores, k // 16, cores, 16, 16)[:, :, core]
        return block.permute(0, 2, 1, 3).reshape(n // cores, k)
    block = physical.view(n // 16, k // 16 // cores, cores, 16, 16)[:, :, core]   # row
    return block.permute(0, 2, 1, 3).reshape(n, k // cores)


@pytest.mark.parametrize("cores", [4, 6])
def test_padded_mlp6_shards_compute_the_unpadded_mlp_on_cpu(cores):
    """Zero gate/up rows and down columns of the 8256 stripe contribute nothing."""
    torch.manual_seed(5)
    hidden, logical = 64, 8192
    physical_size = decoder_mlp_intermediate_size(logical, cores)
    mlp = torch.nn.Module()
    for name, n, k in (("gate_proj", logical, hidden), ("up_proj", logical, hidden),
                       ("down_proj", hidden, logical)):
        mlp.add_module(name, torch.nn.Linear(k, n, bias=False, dtype=torch.float16))
        with torch.no_grad():
            getattr(mlp, name).weight.copy_(torch.randn(n, k) / 16)
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=hidden, intermediate_size=logical),
                            model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]))
    source = {name: child.weight.detach().double() for name, child in mlp.named_children()}
    x = torch.randn(5, hidden, dtype=torch.float64) / 4
    expected = (torch.nn.functional.silu(x @ source["gate_proj"].t()) *
                (x @ source["up_proj"].t())) @ source["down_proj"].t()

    weights.pad_decoder_mlp_weights(model, logical_size=logical, physical_size=physical_size)
    weights.convert_linear_weights_inplace(mlp, attn_num_cores=4, mlp_num_cores=cores,
                                           lm_head_num_cores=4, execution_core_count=cores)
    assert (mlp.gate_proj.out_features, mlp.down_proj.in_features) == (physical_size,) * 2
    for name, child in mlp.named_children():
        assert (child._rpu_linear_partition, child._rpu_linear_num_cores) == (
            0 if name == "down_proj" else 1, cores)
    actual = torch.zeros_like(expected)
    for core in range(cores):
        gate = _shard(mlp.gate_proj.weight.detach(), 1, cores, core).double()
        up = _shard(mlp.up_proj.weight.detach(), 1, cores, core).double()
        down = _shard(mlp.down_proj.weight.detach(), 0, cores, core).double()
        assert gate.shape == (physical_size // cores, hidden)
        actual += (torch.nn.functional.silu(x @ gate.t()) * (x @ up.t())) @ down.t()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    if cores == 6:
        # 8256 = 6 x 1376: the 64 zero channels are the tail of the last shard.
        last = _shard(mlp.gate_proj.weight.detach(), 1, cores, cores - 1)
        assert torch.count_nonzero(last[-64:]) == 0 and torch.count_nonzero(last[:-64]) > 0


@pytest.fixture
def installed(monkeypatch):
    """Run the real LlamaAdapter.to_rpu control flow with device/native steps doubled."""
    from rpu_backend.runtime import hw_attrs
    calls = {}
    for name in ("validate_preinstall", "validate_postinstall", "install_hw_attr_validator"):
        monkeypatch.setattr(hw_attrs, name, lambda *_a, **_k: None)
    monkeypatch.setattr(llama, "require_native_backend", lambda: None)
    monkeypatch.setattr(llama, "_claim_live_instance", lambda _model: None)
    monkeypatch.setattr(llama, "_enable_causal_decoder_execution_reconfigure", lambda _inner: None)
    # The real CPU padding check needs CPU leaves; meta weights keep this test small.
    monkeypatch.setattr(weights, "validate_decoder_mlp_padding",
                        lambda _m, **kw: calls.setdefault("validate_padding", kw))
    monkeypatch.setattr(weights, "pad_decoder_mlp_weights",
                        lambda _m, **kw: calls.setdefault("pad", kw))
    monkeypatch.setattr(llama, "swizzle_model_inplace",
                        lambda _m, **kw: calls.setdefault("swizzle", kw))
    monkeypatch.setattr(LlamaForCausalLM, "to", lambda self, *_a, **_k: self)

    def install(inner, **kwargs):
        calls["install"] = kwargs
        return 17
    monkeypatch.setattr(llama, "_install_causal_decoder_forward", install)
    return calls


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_to_rpu_binds_padding_role_cores_and_installer_topology(installed, cores):
    model = _meta_model(cores)
    adapter = llama.LlamaAdapter(model)
    assert adapter.to_rpu() is model and model._rpu_swizzled
    if cores == 8:
        # Byte-identical eight-core call sequence: no padding, default swizzle,
        # no reduced topology handed to the native installer.
        assert installed["swizzle"] == {"skip_names": set()}
        assert "pad" not in installed and "topology" not in installed["install"]
        assert not hasattr(model, "_rpu_decoder_topology")
        return
    topology = installed["install"]["topology"]
    assert topology.identity() == _TEMPLATES[cores]
    assert model._rpu_decoder_topology is topology
    assert installed["install"]["arch"] == "llama"
    physical = 8256 if cores == 6 else 8192
    assert installed["pad"] == installed["validate_padding"] == {
        "logical_size": 8192, "physical_size": physical}
    assert installed["swizzle"] == {
        "skip_names": set(), "attn_num_cores": 4, "mlp_num_cores": cores,
        "lm_head_num_cores": 4, "execution_core_count": cores}


@pytest.mark.parametrize("cores", [4, 6])
@pytest.mark.parametrize("fault", ["qk_norm", "dtype", "extra_linear", "layer_count"])
def test_actual_reduced_model_is_rejected_before_claim_or_swizzle(installed, cores, fault):
    model = _meta_model(cores)
    attention = model.model.layers[-1].self_attn
    if fault == "qk_norm":
        attention.q_norm = torch.nn.RMSNorm(64, device="meta", dtype=torch.float16)
    elif fault == "dtype":
        model.lm_head.weight = torch.nn.Parameter(torch.empty(128256, 2048, device="meta"))
    elif fault == "extra_linear":
        attention.extra = torch.nn.Linear(2048, 2048, bias=False, device="meta", dtype=torch.float16)
    else:
        del model.model.layers[-1]
    with pytest.raises(UnsupportedModelError):
        llama.LlamaAdapter(model)
    assert not getattr(model, "_rpu_swizzle_started", False) and installed == {}


def test_installer_rejects_reduced_llama_scales_norms_or_layers_before_native(monkeypatch):
    from rpu_backend.runtime.decoder import _install_causal_decoder_forward
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_create",
                        lambda *_a, **_k: pytest.fail("native decoder created"), raising=False)
    topology = resolve_decoder_topology(num_cores=6, **llama_core_profile(_config(), 6).geometry())
    layers = [object()] * 16
    for kwargs, layer_list in (({"scale_lists": ([],) * 7}, layers),
                               ({"qk_norm_lists": ([], [])}, layers), ({}, layers[:15])):
        model = SimpleNamespace(layers=layer_list, config=_config())
        with pytest.raises(ValueError, match="reduced-core Llama install"):
            _install_causal_decoder_forward(
                model, arch="llama", chunk_envelope_for=llama.lookup_causal_decoder,
                execution_config={"model": {"num_cores": 6}}, topology=topology, **kwargs)


@pytest.mark.parametrize("cores", [4, 6])
@pytest.mark.parametrize("installed_topology", [False, True])
def test_cache_uses_attention_tp4_and_eight_physical_kv_stripes(cores, installed_topology):
    config = _config()
    topology = llama._execution_topology(config, _execution(cores))
    model = SimpleNamespace(config=config, _rpu_execution=_execution(cores))
    if installed_topology:
        model._rpu_decoder_topology = topology
    cache = RPUCache.from_model(model, max_seq_len=64, device="cpu")
    validate_decoder_cache_topology(topology, cache)
    assert (cache.attn_tp, cache.physical_kv_cores, cache.num_layers) == (4, 8, 16)
    assert cache.nKVHeadVx * cache.nKVHeadChunk == 16
    # Llama caches are batch-one at every budget.
    with pytest.raises(NotImplementedError, match="batch_size > 1"):
        RPUCache.from_model(model, max_seq_len=64, device="cpu", batch_size=2)


def test_public_loader_keeps_cold_llama_request_without_swizzle(monkeypatch):
    from rpu_backend.api import causal_lm
    for cores in (4, 6, 8):
        model = _meta_model(cores)
        del model._rpu_execution
        monkeypatch.setattr(causal_lm.AutoConfig, "from_pretrained", lambda *_a, **_k: model.config)
        monkeypatch.setattr(causal_lm, "resolve_config_architecture", lambda *_a, **_k: "LlamaForCausalLM")
        monkeypatch.setattr(causal_lm, "get_adapter", lambda _arch: llama.LlamaAdapter)
        monkeypatch.setattr(causal_lm.AutoModelForCausalLM, "from_pretrained", lambda *_a, **_k: model)
        returned = causal_lm.RPUModelForCausalLM.from_pretrained(
            "unused", rpu_execution={"model": {"num_cores": cores}})
        assert returned is model and model._rpu_execution["model"] == {"num_cores": cores}
        assert not getattr(model, "_rpu_swizzle_started", False)
