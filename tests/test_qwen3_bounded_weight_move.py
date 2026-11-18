"""Bounded CPU weight ownership during the existing Qwen3 install transaction."""
import copy
import weakref

import pytest
import torch
from torch import nn

from rpu_backend.runtime import weights


def _small_model(tied=False):
    model = nn.Module()
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(384, 384, dtype=torch.float16)
    model.model.layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.self_attn, layer.mlp = nn.Module(), nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, name, nn.Linear(384, 384, bias=False, dtype=torch.float16))
        for name in ("gate_proj", "up_proj"):
            setattr(layer.mlp, name, nn.Linear(384, 768, bias=False, dtype=torch.float16))
        layer.mlp.down_proj = nn.Linear(768, 384, bias=False, dtype=torch.float16)
        layer.norm = nn.LayerNorm(384, dtype=torch.float16)
        model.model.layers.append(layer)
    model.model.register_buffer("rotary", torch.arange(16, dtype=torch.float32), persistent=False)
    model.lm_head = nn.Linear(384, 384, bias=False, dtype=torch.float16)
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model


@pytest.mark.parametrize("budget,attn,mlp,head", [(4, 4, 4, 4), (6, 4, 6, 4),
                                               (6, 4, 4, 4), (8, 8, 8, 8)])
@pytest.mark.parametrize("tied", [False, True])
def test_bounded_move_preserves_bytes_tags_and_releases_cpu_temporaries(
        monkeypatch, budget, attn, mlp, head, tied):
    baseline = _small_model(tied)
    bounded = copy.deepcopy(baseline)
    kwargs = dict(attn_num_cores=attn, mlp_num_cores=mlp, lm_head_num_cores=head,
                  execution_core_count=budget, skip_names=set())
    baseline_names = {id(module): name for name, module in baseline.named_modules()}
    baseline_order = [name for name, module in baseline.named_modules() if isinstance(module, nn.Linear)]
    selected = []
    select_layout = weights._linear_conversion_layout
    def observe_layout(module, *args):
        selected.append(baseline_names[id(module)])
        return select_layout(module, *args)
    monkeypatch.setattr(weights, "_linear_conversion_layout", observe_layout)
    monkeypatch.setattr(nn.Module, "to", lambda *_a, **_k: pytest.fail("default converter moved a module"))
    expected = weights.convert_linear_weights_inplace(baseline, **kwargs)
    assert selected == baseline_order  # Default-off visits the original registration order.
    monkeypatch.setattr(weights, "_linear_conversion_layout", select_layout)
    embedding_ref = weakref.ref(bounded.model.embed_tokens.weight)
    embedding_value = bounded.model.embed_tokens.weight.detach().clone()
    bounded_names = {id(module): name for name, module in bounded.named_modules()}
    moves, temporary_refs, previous_parameter_refs = [], [], []
    transform = weights.transform_linear_weight

    def observe_transform(*args, **kwargs):
        result = transform(*args, **kwargs)
        temporary_refs.append(weakref.ref(result))
        return result

    def move(module, target):
        assert isinstance(module, (nn.Linear, nn.Embedding)) and target == torch.device("cpu")
        assert torch.equal(bounded.model.embed_tokens.weight, embedding_value)
        if isinstance(module, nn.Linear):
            assert hasattr(module, "_rpu_linear_partition")
            if module is bounded.lm_head:
                assert bounded.model.embed_tokens.weight is embedding_ref()
                assert module.weight is not embedding_ref()
        else:
            assert moves == [bounded.lm_head]
            assert not hasattr(module, "_rpu_linear_partition")
        # Simulate a different-backend Module.to Parameter replacement using
        # real CPU tensors. This exposes extra references to old CPU weights.
        previous_parameter_refs.append(weakref.ref(module.weight))
        module.weight = nn.Parameter(module.weight.detach().clone())
        moves.append(module)
        return module

    def reclaim():
        # Refcount release is immediate; a collection here would hide an
        # avoidable Python retention and repeatedly scan the pytest process.
        assert temporary_refs[-1]() is None, "swizzle temporary survived the move"
        assert previous_parameter_refs[-1]() is None, "old CPU Parameter was retained"

    monkeypatch.setattr(weights, "transform_linear_weight", observe_transform)
    monkeypatch.setattr(nn.Module, "to", move)
    monkeypatch.setattr(weights, "_release_cpu_weight_pages", reclaim)
    actual = weights.convert_linear_weights_inplace(bounded, move_to_device="cpu", **kwargs)
    assert actual == expected
    assert [bounded_names[id(module)] for module in moves] == [
        "lm_head", "model.embed_tokens", *[name for name in baseline_order if name != "lm_head"]]
    assert embedding_ref() is None
    assert bounded.lm_head.weight is not bounded.model.embed_tokens.weight
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, bounded.state_dict()[name]), name
    for name, module in baseline.named_modules():
        if isinstance(module, nn.Linear):
            counterpart = bounded.get_submodule(name)
            assert (counterpart._rpu_linear_partition, counterpart._rpu_linear_num_cores) == (
                module._rpu_linear_partition, module._rpu_linear_num_cores)
    # The final whole-model migration must not create second copies of already
    # resident weights. Module._apply is the implementation used by Module.to.
    pointers = [module.weight.data_ptr() for module in moves]
    bounded._apply(lambda tensor: tensor.to("cpu"))
    assert pointers == [module.weight.data_ptr() for module in moves]
    assert torch.equal(bounded.model.rotary, baseline.model.rotary)


@pytest.mark.parametrize("fault", ["dtype", "role", "row_geometry", "shape",
                                  "embedding_dtype", "embedding_noncontiguous", "embedding_meta"])
def test_bad_last_linear_rejects_before_any_swizzle_or_move(monkeypatch, fault):
    model = _small_model()
    if fault == "dtype":
        model.lm_head.float()
    elif fault == "role":
        model.unmodelled = nn.Linear(384, 384, bias=False, dtype=torch.float16)
    elif fault == "row_geometry":
        model.model.layers[-1].mlp.down_proj = nn.Linear(770, 384, bias=False, dtype=torch.float16)
    elif fault == "embedding_dtype":
        model.model.embed_tokens.float()
    elif fault == "embedding_noncontiguous":
        model.model.embed_tokens.weight = nn.Parameter(model.model.embed_tokens.weight.detach().t())
    elif fault == "embedding_meta":
        model.model.embed_tokens.weight = nn.Parameter(torch.empty(384, 384, device="meta", dtype=torch.float16))
    else:
        model.lm_head.weight = nn.Parameter(torch.zeros(383, 384, dtype=torch.float16))
    monkeypatch.setattr(nn.Module, "to", lambda *_a, **_k: pytest.fail("bad tree moved a weight"))
    monkeypatch.setattr(weights, "transform_linear_weight", lambda *_a, **_k: pytest.fail("bad tree swizzled a weight"))
    with pytest.raises(ValueError, match="bounded weight migration"):
        weights.convert_linear_weights_inplace(model, move_to_device="cpu")
    assert not hasattr(model.model.layers[0].self_attn.q_proj, "_rpu_linear_partition")


def _meta_hf(size, cores):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    dims = {"0.6b": (1024, 3072, 28, 16), "1.7b": (2048, 6144, 28, 16),
            "4b": (2560, 9728, 36, 32), "8b": (4096, 12288, 36, 32)}
    hidden, intermediate, layers, heads = dims[size]
    from torch._subclasses.fake_tensor import FakeTensorMode
    # Exact8B preflight checks CPU residency before claiming a device model.
    # FakeTensor uses that real metadata without allocating multi-GB storage.
    with (FakeTensorMode() if size == "8b" or (size == "4b" and cores == 6) else torch.device("meta")):
        model = Qwen3ForCausalLM._from_config(Qwen3Config(hidden_size=hidden,
            intermediate_size=intermediate, num_hidden_layers=layers,
            num_attention_heads=heads, num_key_value_heads=8, head_dim=128,
            vocab_size=151936), dtype=torch.float16)
    model._rpu_execution = {"model": {"num_cores": cores}}
    return model


@pytest.mark.parametrize("size,cores", [(size, cores) for size in ("0.6b", "1.7b", "4b", "8b")
                                       for cores in (6, 8)] + [("8b", 4)])
def test_real_adapter_selects_only_8b_and_keeps_failed_install_poisoned(monkeypatch, size, cores):
    from rpu_backend.adapters import qwen3
    model = _meta_hf(size, cores)
    adapter = qwen3.Qwen3Adapter(model)
    claims, calls = [], []
    monkeypatch.setattr(qwen3, "_claim_live_instance", lambda owner: claims.append(id(owner)))
    monkeypatch.setattr(qwen3, "require_native_backend", lambda: None)

    def fail_move(owner, **kwargs):
        calls.append(kwargs)
        assert owner._rpu_swizzle_started
        raise RuntimeError("injected device move failure")

    monkeypatch.setattr(qwen3, "convert_linear_weights_inplace", fail_move)
    with pytest.raises(RuntimeError, match="injected device move failure"):
        adapter.to_rpu()
    assert calls[0].get("move_to_device") == ("rpu" if size == "8b" else None)
    assert len(claims) == len(calls) == 1
    assert model._rpu_swizzle_started and not getattr(model, "_rpu_swizzled", False)
    assert not adapter._rpu_is_ready and adapter._all_layers_once_handle is None
    assert not hasattr(model.model, "_rpu_decoder_handle")
    with pytest.raises(qwen3.RPUBackendError, match="prior swizzle attempt"):
        adapter.to_rpu()
    assert len(claims) == len(calls) == 1


@pytest.mark.parametrize("fault", ["last_norm_dtype", "unknown_linear", "profile_drift",
                                  "last_noncontiguous", "last_meta",
                                  "embedding_noncontiguous", "embedding_meta"])
def test_eight_core_8b_checks_actual_tree_before_claim(monkeypatch, fault):
    from rpu_backend.adapters import qwen3
    model = _meta_hf("8b", 8)
    adapter = qwen3.Qwen3Adapter(model)
    if fault == "last_norm_dtype":
        norm = model.model.layers[-1].post_attention_layernorm
        with norm.weight.fake_mode:
            norm.weight = nn.Parameter(torch.empty(norm.weight.shape, dtype=torch.float32))
    elif fault == "profile_drift":
        model.config.use_sliding_window = True
    elif fault in {"last_noncontiguous", "embedding_noncontiguous"}:
        owner = model.lm_head if fault == "last_noncontiguous" else model.model.embed_tokens
        original = owner.weight
        with original.fake_mode:
            owner.weight = nn.Parameter(torch.empty(
                original.shape[1], original.shape[0], dtype=torch.float16).t())
    elif fault in {"last_meta", "embedding_meta"}:
        owner = model.lm_head if fault == "last_meta" else model.model.embed_tokens
        owner.weight = nn.Parameter(torch.empty(owner.weight.shape, dtype=torch.float16, device="meta"))
    else:
        with torch.device("meta"):
            model.extra = nn.Linear(4096, 4096, bias=False, dtype=torch.float16)
    monkeypatch.setattr(qwen3, "_claim_live_instance", lambda *_: pytest.fail("invalid tree claimed model"))
    with pytest.raises((qwen3.UnsupportedModelError, ValueError), match="decoder|bounded"):
        adapter.to_rpu()
    assert not getattr(model, "_rpu_swizzle_started", False)


@pytest.mark.parametrize("fail_at", ["head", "embedding"])
def test_real_adapter_large_weight_move_failure_keeps_transaction_poisoned(monkeypatch, fail_at):
    from rpu_backend.adapters import qwen3
    model = _meta_hf("8b", 4)
    adapter = qwen3.Qwen3Adapter(model)
    claims, moves = [], []
    monkeypatch.setattr(qwen3, "_claim_live_instance", lambda owner: claims.append(id(owner)))
    monkeypatch.setattr(qwen3, "require_native_backend", lambda: None)
    # Give the metadata fixture stable object identities without reading fake
    # storage. Real CPU tied/untied ownership is checked by the test above.
    from torch._subclasses.fake_tensor import FakeTensor
    monkeypatch.setattr(FakeTensor, "data_ptr", lambda tensor: id(tensor))
    monkeypatch.setattr(weights, "_release_cpu_weight_pages", lambda: None)
    def move(module, target):
        assert target == torch.device("rpu") and model._rpu_swizzle_started
        role = "head" if module is model.lm_head else "embedding"
        assert module is model.lm_head or module is model.model.embed_tokens
        moves.append(role)
        if role == fail_at:
            raise RuntimeError(f"injected {role} allocation failure")
        with module.weight.fake_mode:
            module.weight = nn.Parameter(module.weight.detach().clone())
        return module
    monkeypatch.setattr(nn.Module, "to", move)
    with pytest.raises(RuntimeError, match=f"injected {fail_at} allocation failure"):
        adapter.to_rpu()
    assert moves == (["head"] if fail_at == "head" else ["head", "embedding"])
    assert claims == [id(model)] and model._rpu_swizzle_started
    assert not adapter._rpu_is_ready and adapter._all_layers_once_handle is None
    assert not hasattr(model.model, "_rpu_decoder_handle")
    with pytest.raises(qwen3.RPUBackendError, match="prior swizzle attempt"):
        adapter.to_rpu()
    assert claims == [id(model)]
