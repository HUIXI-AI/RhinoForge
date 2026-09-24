"""Bounded CPU weight staging without loading a checkpoint or native SDK."""
from types import SimpleNamespace
import weakref

import pytest
import torch

from rpu_backend.adapters import qwen3_vl as qvl
from rpu_backend.runtime.weights import (
    convert_linear_weights_inplace, transform_linear_weight,
)
from rpu_backend.tests.acceptance.test_qwen3_vl_install_lifecycle import (  # noqa: F401
    install_runtime, _Model,
)


@pytest.mark.parametrize("fail_transfer", [False, True])
def test_8b_conversion_releases_each_layer_before_next_and_cannot_retry(
    install_runtime, monkeypatch, fail_transfer,
):
    model = _Model()
    # Exercise the 8B installation route with tiny weights; config admission is
    # covered separately and does not need an 8B allocation for this contract.
    model.config.text_config.hidden_size = 4096
    model.config.text_config.intermediate_size = 12288
    model.config.text_config.num_hidden_layers = 36
    monkeypatch.setattr(qvl, "_check_profile", lambda config: None)
    pending = set()
    conversions = []
    expected = []
    roles = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

    class Layer(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index
            self.self_attn = torch.nn.Module()
            self.mlp = torch.nn.Module()
            reference = {}
            for role in roles:
                owner, name = role.split(".")
                projection = torch.nn.Linear(128, 128, bias=False).half()
                with torch.no_grad():
                    projection.weight.copy_(torch.arange(128 * 128).reshape(128, 128) % 37)
                getattr(self, owner).add_module(name, projection)
                reference[role] = transform_linear_weight(
                    projection.weight.detach(), 0 if name in ("o_proj", "down_proj") else 1)
            expected.append(reference)

        def to(self, device):
            assert device == "rpu" and pending == {self.index}
            if fail_transfer and self.index == 1:
                raise RuntimeError("layer transfer failed")
            # Projection Parameters already point into the per-layer bank.
            assert len({self.get_submodule(role).weight.untyped_storage().data_ptr()
                        for role in roles}) == 1
            pending.remove(self.index)
            return self

    layers = [Layer(index) for index in range(3)]
    model.model.language_model.layers = layers
    def get_submodule(name):
        if name == "model.language_model":
            return model.model.language_model
        prefix, relative = name.split(".layers.")
        index, role = relative.split(".", 1)
        assert prefix == "model.language_model"
        return layers[int(index)].get_submodule(role)
    model.get_submodule = get_submodule
    model.get_parameter = lambda name: get_submodule(name.removesuffix(".weight")).weight
    empty = torch.empty
    def allocate(*args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return empty(*args, **kwargs)
    monkeypatch.setattr(torch, "empty", allocate)

    def convert(layer, *args, **kwargs):
        if not isinstance(layer, Layer):
            return convert_linear_weights_inplace(layer, *args, **kwargs)
        assert not pending, "previous converted CPU layer is still waiting for transfer"
        conversions.append(layer.index)
        pending.add(layer.index)
        return convert_linear_weights_inplace(layer, **kwargs)

    from rpu_backend.runtime import weights
    monkeypatch.setattr(weights, "convert_linear_weights_inplace", convert)
    adapter = qvl.Qwen3VLAdapter(model)
    if fail_transfer:
        with pytest.raises(RuntimeError, match="layer transfer failed"):
            adapter.to_rpu()
        assert conversions == [0, 1]
        with pytest.raises(qvl.RPUBackendError, match="prior swizzle attempt|closed"):
            adapter.to_rpu()
        assert conversions == [0, 1]
    else:
        assert adapter.to_rpu() is model
        assert conversions == [0, 1, 2] and not pending
        for layer, reference in zip(layers, expected):
            for role, value in reference.items():
                assert layer.get_submodule(role).weight.dtype == torch.float16
                assert torch.equal(layer.get_submodule(role).weight, value)
        assert adapter.to_rpu() is model
        assert conversions == [0, 1, 2]
        adapter.close()


def test_lm_head_cpu_temporaries_are_released_before_vision_install(
    install_runtime, monkeypatch,
):
    model = _Model()
    temporary_refs = []

    def transform(weight, partition):
        result = weight.clone()
        temporary_refs.extend((weakref.ref(weight), weakref.ref(result)))
        return result

    monkeypatch.setattr(qvl, "transform_linear_weight", transform)
    install_vision = qvl.install_qwen3_vl_vision_for_rpu

    def vision(child, **kwargs):
        assert len(temporary_refs) == 2
        assert all(reference() is None for reference in temporary_refs)
        return install_vision(child, **kwargs)

    monkeypatch.setattr(qvl, "install_qwen3_vl_vision_for_rpu", vision)
    adapter = qvl.Qwen3VLAdapter(model)
    adapter.to_rpu()
    adapter.close()
