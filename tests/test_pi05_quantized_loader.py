"""Exercise the complete small safetensors loader with mixed W4-G32/KV8 data."""
import json
import sys
from types import ModuleType, SimpleNamespace

import torch
from safetensors.torch import save_file

from rpu_backend.adapters.pi05.loader import _load_and_fp16_cast
from rpu_backend.quant.convert_pi05 import _convert_tensor, _quant_config


def test_loader_passes_loaded_group_config_and_preserves_mixed_weights(tmp_path, monkeypatch):
    class TinyPI05(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config.device is None and not config.compile_model
            self.model = torch.nn.Module()
            self.model.paligemma_with_expert = torch.nn.Module()
            pawe = self.model.paligemma_with_expert
            pawe.paligemma = torch.nn.Module()
            pawe.paligemma.model = torch.nn.Module()
            pawe.paligemma.model.language_model = torch.nn.Module()
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.q_proj = torch.nn.Linear(64, 128, bias=False)
            layer.self_attn.k_proj = torch.nn.Linear(64, 128, bias=False)
            pawe.paligemma.model.language_model.layers = torch.nn.ModuleList([layer])

    # Only the external architecture factory is doubled. JSON, safetensors,
    # meta construction, quantized extraction, load_state_dict and install run.
    modules = {name: ModuleType(name) for name in (
        "lerobot", "lerobot.policies", "lerobot.policies.pi05",
        "lerobot.policies.pi05.modeling_pi05", "lerobot.configs",
        "lerobot.configs.policies",
    )}
    modules["lerobot.policies.pi05.modeling_pi05"].PI05Policy = TinyPI05
    modules["lerobot.configs.policies"].PreTrainedConfig = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(device="cpu"),
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    config = _quant_config(bits=4, real_w4=True)
    (tmp_path / "rpu_quant_config.json").write_text(json.dumps(config))
    checkpoint = {}
    prefix = "model.paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn."
    for name in ("q_proj", "k_proj"):
        converted, *_ = _convert_tensor(
            prefix + name + ".weight",
            torch.linspace(-2, 2, 128 * 64).reshape(128, 64),
            bits=4, group_size=32,
            int8_keep_suffixes=tuple(config["mixed_int8_suffixes"]),
        )
        checkpoint.update(converted)
    save_file(checkpoint, tmp_path / "model_remapped.safetensors")
    original_init = TinyPI05.__init__

    policy = _load_and_fp16_cast(str(tmp_path), torch.float16)

    assert TinyPI05.__init__ is original_init
    assert not policy.training and policy._pi05_quant_config == config
    projections = policy.model.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn
    for name, expected_scale_shape in (("q_proj", (2, 128)), ("k_proj", (128,))):
        projection = getattr(projections, name)
        assert projection.weight.dtype == torch.int8
        assert projection.weight_scale.dtype == torch.float16
        assert tuple(projection.weight_scale.shape) == expected_scale_shape
        assert torch.equal(projection.weight, checkpoint[prefix + name + ".weight"])
        assert torch.equal(projection.weight_scale, checkpoint[prefix + name + ".weight_scale"])
    assert all(p.device.type == "cpu" for p in policy.parameters())
    assert all(b.device.type == "cpu" for b in policy.buffers())
