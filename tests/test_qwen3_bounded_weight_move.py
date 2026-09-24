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


@pytest.mark.parametrize("tied", [False, True])
def test_bounded_move_preserves_bytes_tags_and_releases_cpu_temporaries(
        monkeypatch, tied):
    baseline = _small_model(tied)
    bounded = copy.deepcopy(baseline)
    kwargs = dict(skip_names=set())
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


@pytest.mark.parametrize("fault", ["dtype", "role", "row_geometry", "embedding_noncontiguous"])
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


def _meta_hf(size="8b", w8=False):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from torch._subclasses.fake_tensor import FakeTensorMode
    hidden, intermediate, layers, heads = {
        "0.6b": (1024, 3072, 28, 16), "8b": (4096, 12288, 36, 32),
    }[size]
    with FakeTensorMode():
        model = Qwen3ForCausalLM._from_config(Qwen3Config(hidden_size=hidden,
            intermediate_size=intermediate, num_hidden_layers=layers,
            num_attention_heads=heads, num_key_value_heads=8, head_dim=128,
            vocab_size=151936), dtype=torch.float16)
        if w8:
            for layer in model.model.layers:
                for owner, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                                      (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
                    for name in names:
                        projection = getattr(owner, name)
                        projection.weight = nn.Parameter(projection.weight.to(torch.int8), requires_grad=False)
                        projection.register_buffer("weight_scale", torch.ones(projection.out_features, dtype=torch.float16))
            model.config.quant_config = {"method": "w8a16"}
    return model


@pytest.mark.parametrize("size,w8", [("0.6b", False), ("8b", False), ("8b", True)])
def test_adapter_routes_only_plain_8b_to_bounded_move(monkeypatch, size, w8):
    from rpu_backend.adapters import qwen3
    model = _meta_hf(size, w8)
    adapter = qwen3.Qwen3Adapter(model)
    calls = []
    monkeypatch.setattr(qwen3, "_claim_live_instance", lambda _model: None)

    def fail_move(owner, **kwargs):
        calls.append((owner, kwargs))
        assert model._rpu_swizzle_started
        raise RuntimeError("injected device move failure")

    monkeypatch.setattr(qwen3, "convert_linear_weights_inplace", fail_move)
    with pytest.raises(RuntimeError, match="injected device move failure"):
        adapter.to_rpu()
    assert calls[0][1].get("move_to_device") == ("rpu" if size == "8b" and not w8 else None)
    assert calls[0][0] is (model.model.layers[0] if w8 else model)
    assert model._rpu_swizzle_started and not getattr(model, "_rpu_swizzled", False)
    assert not adapter._rpu_is_ready and not hasattr(model.model, "_rpu_decoder_handle")
    with pytest.raises(qwen3.RPUBackendError, match="prior swizzle attempt"):
        adapter.to_rpu()
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ["last_norm_dtype", "unknown_linear", "profile_drift", "last_noncontiguous"])
def test_actual_8b_tree_is_checked_before_claim(monkeypatch, fault):
    from rpu_backend.adapters import qwen3
    model = _meta_hf()
    adapter = qwen3.Qwen3Adapter(model)
    if fault == "last_norm_dtype":
        norm = model.model.layers[-1].post_attention_layernorm
        with norm.weight.fake_mode:
            norm.weight = nn.Parameter(torch.empty(norm.weight.shape, dtype=torch.float32))
    elif fault == "profile_drift":
        model.config.use_sliding_window = True
    elif fault == "last_noncontiguous":
        original = model.lm_head.weight
        with original.fake_mode:
            model.lm_head.weight = nn.Parameter(torch.empty(
                original.shape[1], original.shape[0], dtype=torch.float16).t())
    else:
        with torch.device("meta"):
            model.extra = nn.Linear(4096, 4096, bias=False, dtype=torch.float16)
    monkeypatch.setattr(qwen3, "_claim_live_instance", lambda *_: pytest.fail("invalid tree claimed model"))
    with pytest.raises((qwen3.UnsupportedModelError, ValueError), match="decoder|bounded"):
        adapter.to_rpu()
    assert not getattr(model, "_rpu_swizzle_started", False)


def test_late_upload_failure_poison_preserves_completed_moves(monkeypatch):
    from rpu_backend.adapters import qwen3
    from torch._subclasses.fake_tensor import FakeTensor
    model = _meta_hf()
    adapter = qwen3.Qwen3Adapter(model)
    claims, moves = [], []
    monkeypatch.setattr(qwen3, "_claim_live_instance", lambda owner: claims.append(id(owner)))
    monkeypatch.setattr(FakeTensor, "data_ptr", lambda tensor: id(tensor))
    monkeypatch.setattr(weights, "_release_cpu_weight_pages", lambda: None)
    names = {id(module): name for name, module in model.named_modules()}
    last = model.model.layers[-1].mlp.down_proj

    def move(module, target):
        assert target == torch.device("rpu") and model._rpu_swizzle_started
        moves.append(names[id(module)])
        if module is last:
            raise RuntimeError("injected late allocation failure")
        with module.weight.fake_mode:
            module.weight = nn.Parameter(module.weight.detach().clone())
        return module

    monkeypatch.setattr(nn.Module, "to", move)
    with pytest.raises(RuntimeError, match="injected late allocation failure"):
        adapter.to_rpu()
    assert moves[:2] == ["lm_head", "model.embed_tokens"]
    assert moves[-1] == "model.layers.35.mlp.down_proj" and len(moves) == 254
    assert claims == [id(model)] and model._rpu_swizzle_started
    assert not adapter._rpu_is_ready and not hasattr(model.model, "_rpu_decoder_handle")
    with pytest.raises(qwen3.RPUBackendError, match="prior swizzle attempt"):
        adapter.to_rpu()
    assert claims == [id(model)] and len(moves) == 254
