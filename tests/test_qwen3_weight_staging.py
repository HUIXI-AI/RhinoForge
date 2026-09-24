"""Bound CPU conversion storage using real FP16/INT8 swizzles and tiny layers."""
import pytest
import torch

from rpu_backend.adapters import qwen3
from rpu_backend.runtime import decoder
from rpu_backend.runtime.weights import (
    convert_linear_weights_inplace, transform_linear_weight,
)
from test_qwen3_fused_lm_head_install import _Model, runtime  # noqa: F401


@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("fail_transfer", [False, True])
def test_large_qwen3_stages_each_layer_once_and_releases_cpu_storage(
    runtime, monkeypatch, int8, fail_transfer,
):
    # Use the real 8B installation branch with tiny storage. Admission itself
    # is covered elsewhere; no checkpoint, model forward or device is loaded.
    model = _Model(8, int8_head=False, int8_decoder=int8)
    # The remainder converter needs a real swizzlable vocabulary head.
    model.lm_head = torch.nn.Linear(128, 128, bias=False).half()
    conversions = []
    pending = set()
    storage_refs = []
    expected = {}

    class Layer(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index
            self.self_attn = torch.nn.Module()
            self.mlp = torch.nn.Module()
            for parent, names in (
                (self.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                (self.mlp, ("gate_proj", "up_proj", "down_proj")),
            ):
                for name in names:
                    linear = torch.nn.Linear(256, 256, bias=False)
                    values = (torch.arange(256 * 256).reshape(256, 256) % 37 - 18)
                    linear.weight = torch.nn.Parameter(
                        values.to(torch.int8 if int8 else torch.float16),
                        requires_grad=False,
                    )
                    if int8:
                        linear.register_buffer("weight_scale", torch.ones(256, dtype=torch.float16))
                    setattr(parent, name, linear)
                    partition = 0 if name in ("o_proj", "down_proj") else 1
                    expected[index, name] = transform_linear_weight(linear.weight.detach(), partition)

        def to(self, device):
            assert device == "rpu" and pending == {self.index}
            if fail_transfer and self.index == 1:
                raise RuntimeError("layer transfer failed")
            # Stand in only for transfer: replace storage like Module.to('rpu').
            # Weak storage handles below prove the converted CPU allocation is
            # actually gone, independently of the lifetime of Tensor wrappers.
            for parameter in self.parameters():
                parameter.data = parameter.data.clone()
            pending.remove(self.index)
            return self

    model.model.layers = torch.nn.ModuleList([Layer(i) for i in range(3)])
    original_layers = model.model.layers
    raw_embedding = model.model.embed_tokens.weight.detach().clone()

    def convert(module, **kwargs):
        assert not pending, "previous CPU layer is still waiting for transfer"
        assert all(torch.UntypedStorage._expired(ref) for ref in storage_refs)
        if isinstance(module, Layer):
            conversions.append(module.index)
            pending.add(module.index)
        else:
            # The remainder walk must keep the original model tree registered
            # while excluding every already-converted decoder projection.
            assert model.model.layers is original_layers
            assert all(layer not in set(module.modules()) for layer in original_layers)
            conversions.append("remaining")
        result = convert_linear_weights_inplace(module, **kwargs)
        if isinstance(module, Layer):
            storage_refs.extend(p.untyped_storage()._weak_ref() for p in module.parameters())
        return result

    monkeypatch.setattr(qwen3, "convert_linear_weights_inplace", convert)
    adapter = qwen3.Qwen3Adapter(model)
    try:
        if fail_transfer:
            with pytest.raises(RuntimeError, match="layer transfer failed"):
                adapter.to_rpu()
            assert conversions == [0, 1]
            assert "install" not in runtime["events"]
            assert model._rpu_swizzle_started and not adapter._rpu_is_ready
            with pytest.raises(qwen3.RPUBackendError, match="prior swizzle attempt"):
                adapter.to_rpu()
            assert conversions == [0, 1]
        else:
            assert adapter.to_rpu() is model
            assert conversions == [0, 1, 2, "remaining"] and not pending
            for layer in original_layers:
                for name, projection in layer.named_modules():
                    if isinstance(projection, torch.nn.Linear):
                        assert torch.equal(projection.weight, expected[layer.index, name.rsplit(".", 1)[-1]])
                        if int8:
                            assert projection.weight.dtype == torch.int8
                            assert torch.equal(projection.weight_scale, torch.ones(256, dtype=torch.float16))
            assert torch.equal(model.model.embed_tokens.weight, raw_embedding)
            assert adapter.to_rpu() is model
            assert conversions == [0, 1, 2, "remaining"]
            decoder._close_causal_lm_model(model)
    finally:
        for ref in storage_refs:
            torch.UntypedStorage._free_weak_ref(ref)


def test_remaining_tree_preserves_tied_embedding_and_converts_other_linear_children():
    model = _Model(8, int8_head=False, int8_decoder=False)
    model.model.layers = torch.nn.ModuleList()
    model.model.embed_tokens = torch.nn.Embedding(128, 128).half()
    model.lm_head = torch.nn.Linear(128, 128, bias=False).half()
    model.lm_head.weight = model.model.embed_tokens.weight
    model.model.extra_projection = torch.nn.Linear(128, 128, bias=False).half()
    embedding = model.model.embed_tokens.weight.detach().clone()
    extra = transform_linear_weight(model.model.extra_projection.weight.detach(), 1)

    qwen3._stage_large_qwen3_weights_for_rpu(model)

    assert model.model.embed_tokens.weight is not model.lm_head.weight
    assert torch.equal(model.model.embed_tokens.weight, embedding)
    assert torch.equal(model.lm_head.weight, transform_linear_weight(embedding, 1))
    assert torch.equal(model.model.extra_projection.weight, extra)
