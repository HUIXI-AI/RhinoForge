"""Host loading and rejection boundaries for the exact pre-v1 27B text bundle."""
from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from rpu_backend.adapters.qwen3_5 import loader, quant_scope as qs
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError
from rpu_backend.quant.convert_qwen3_5 import convert_checkpoint


LEGACY = {
    "method": "w8a16", "mode": "per_channel_symmetric", "qaxis": 0,
    "skip_modules": ["lm_head"], "quantized_lm_head": False,
    "quantized_embed_tokens": False, "lm_head_untied": False,
    "embed_tokens_untied": False,
}
LAYER_TYPES = ([qs.LINEAR_ATTENTION] * 3 + [qs.FULL_ATTENTION]) * 16


def config(quant=None):
    text = SimpleNamespace(
        hidden_size=5120, intermediate_size=17408, num_hidden_layers=64,
        num_attention_heads=24, num_key_value_heads=4, head_dim=256,
        linear_num_key_heads=16, linear_num_value_heads=48,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, layer_types=list(LAYER_TYPES),
        rope_parameters={"rope_theta": 10000000, "partial_rotary_factor": 0.25,
                         "mrope_section": [11, 11, 10], "mrope_interleaved": True,
                         "rope_type": "default"},
    )
    return SimpleNamespace(text_config=text, quant_config=deepcopy(LEGACY if quant is None else quant))


class TinyTextModel(nn.Module):
    """Real Parameters/state loading, with tiny projection shapes and no HF weights."""

    def __init__(self, text_config):
        super().__init__()
        self.config = text_config
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module() for _ in LAYER_TYPES])
        for relpath in qs.quant_relpaths(qs.is_full_from_layer_types(LAYER_TYPES)):
            _, index, family, projection = relpath.split(".")
            layer = self.model.layers[int(index)]
            if not hasattr(layer, family):
                layer.add_module(family, nn.Module())
            getattr(layer, family).add_module(projection, nn.Linear(2, 2, bias=False))
        self.model.embed_tokens = nn.Embedding(3, 2)
        self.model.norm = nn.LayerNorm(2)
        self.lm_head = nn.Linear(2, 3, bias=False)

    def tie_weights(self):
        pass


@pytest.fixture(scope="module")
def tensors():
    model = TinyTextModel(config().text_config)
    specs = qs.resolve_quant_specs(qs.is_full_from_layer_types(LAYER_TYPES), LEGACY)
    state = {}
    for key, value in model.state_dict().items():
        raw = "model.language_model." + key[len("model."):] if key.startswith("model.") else key
        spec = specs.get(key.removeprefix("model.").removesuffix(".weight"))
        if spec is not None and spec.method == qs.W8A16:
            state[raw] = torch.arange(value.numel(), dtype=torch.int8).reshape(value.shape)
            state[qs.scale_key(raw)] = torch.full((value.size(0),), 0.25, dtype=torch.float16)
        else:
            state[raw] = torch.ones_like(value, dtype=torch.float16)
    state["model.visual.blocks.0.weight"] = torch.ones(2, dtype=torch.float16)
    state["mtp.layers.0.mlp.gate_proj.weight"] = torch.ones(2, 2, dtype=torch.int8)
    state["mtp.layers.0.mlp.gate_proj.weight_scale"] = torch.ones(2, dtype=torch.float16)
    return state


@pytest.fixture
def text_loader(monkeypatch):
    from transformers import AutoModelForCausalLM

    calls = []

    def construct(text_config):
        calls.append(text_config)
        return TinyTextModel(text_config)

    monkeypatch.setattr(AutoModelForCausalLM, "from_config", construct)
    monkeypatch.setattr(loader.AutoModelForImageTextToText, "from_config",
                        lambda *_a, **_k: pytest.fail("legacy must not instantiate Vision"))
    return calls


def write_shards(path, shards):
    weight_map = {}
    for i, state in enumerate(shards):
        name = f"model-{i:05d}.safetensors"
        save_file(state, path / name)
        weight_map.update({key: name for key in state})
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


@pytest.mark.parametrize("split_scales", [False, True])
def test_public_loader_keeps_int8_scales_and_skips_vision_mtp(
    tmp_path, tensors, text_loader, monkeypatch, split_scales
):
    if split_scales:
        write_shards(tmp_path, [
            {k: v for k, v in tensors.items() if not k.endswith("_scale")},
            {k: v for k, v in tensors.items() if k.endswith("_scale")},
        ])
    else:
        save_file(tensors, tmp_path / "model.safetensors")
    read_keys = []
    real_open = loader.safe_open

    class TracedOpen:
        def __init__(self, *args, **kwargs):
            self.raw = real_open(*args, **kwargs)

        def __enter__(self):
            self.raw.__enter__()
            return self

        def __exit__(self, *args):
            return self.raw.__exit__(*args)

        def keys(self):
            return self.raw.keys()

        def get_tensor(self, key):
            read_keys.append(key)
            return self.raw.get_tensor(key)

    monkeypatch.setattr(loader, "safe_open", TracedOpen)
    cfg = config()
    model = loader.load_qwen3_5_model(tmp_path, config=cfg)
    assert text_loader == [cfg.text_config]
    assert not any(k.startswith(("model.visual.", "mtp.")) for k in read_keys)
    assert not any(p.requires_grad or p.device.type == "meta" for p in model.parameters())
    assert not model.training
    for relpath, spec in qs.resolve_quant_specs(qs.is_full_from_layer_types(LAYER_TYPES), LEGACY).items():
        module = model.get_submodule("model." + relpath)
        raw = qs.TEXT_PREFIX + relpath + ".weight"
        assert torch.equal(module.weight, tensors[raw])
        if spec.method == qs.W8A16:
            assert module.weight.dtype == torch.int8
            assert torch.equal(module.weight_scale, tensors[qs.scale_key(raw)])
        else:
            assert module.weight.dtype == torch.float16
            assert not hasattr(module, "weight_scale")
    assert model._qwen3_5_legacy_text_only
    assert qs.validate_legacy_text_profile(model.config)
    assert model.config.quant_config == model._qwen3_5_quant_config


@pytest.mark.parametrize("key", [
    "model.language_model.layers.0.mlp.typo.weight",
    "model.language_model.embed_tokens.weight_scale", "lm_head.weight_scale",
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale",
    "unrecognized.weight",
])
def test_unknown_text_weights_or_scales_fail_closed(tmp_path, tensors, text_loader, key):
    save_file(dict(tensors, **{key: torch.ones(2, dtype=torch.float16)}), tmp_path / "model.safetensors")
    with pytest.raises(RPUBackendError, match="out-of-scope text tensor"):
        loader.load_qwen3_5_model(tmp_path, config=config())


@pytest.mark.parametrize("key", [
    "lm_head.weight", "model.language_model.layers.0.mlp.gate_proj.weight_scale",
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
])
def test_missing_text_parameter_or_scale_fails_closed(tmp_path, tensors, text_loader, key):
    save_file({k: v for k, v in tensors.items() if k != key}, tmp_path / "model.safetensors")
    with pytest.raises(RPUBackendError, match="missing text tensor"):
        loader.load_qwen3_5_model(tmp_path, config=config())


@pytest.mark.parametrize("alias", [False, True])
def test_shards_cannot_overwrite_an_existing_weight(tmp_path, tensors, text_loader, alias):
    key = "model.language_model.layers.0.mlp.gate_proj.weight"
    duplicate = key.replace("model.language_model.", "model.") if alias else key
    write_shards(tmp_path, [tensors, {duplicate: tensors[key]}])
    with pytest.raises(RPUBackendError, match="duplicate"):
        loader.load_qwen3_5_model(tmp_path, config=config())


def test_shard_index_must_match_actual_tensor_location(tmp_path, tensors, text_loader):
    write_shards(tmp_path, [tensors])
    index = tmp_path / "model.safetensors.index.json"
    value = json.loads(index.read_text())
    del value["weight_map"]["lm_head.weight"]
    index.write_text(json.dumps(value))
    with pytest.raises(RPUBackendError, match="disagree with the weight_map"):
        loader.load_qwen3_5_model(tmp_path, config=config())


def test_ordinary_int8_parameter_cannot_escape_the_declared_scope(tmp_path, tensors, text_loader):
    save_file(dict(tensors, **{"lm_head.weight": torch.ones(3, 2, dtype=torch.int8)}),
              tmp_path / "model.safetensors")
    with pytest.raises(RPUBackendError, match="ordinary parameter .* must be floating point"):
        loader.load_qwen3_5_model(tmp_path, config=config())


@pytest.mark.parametrize("field,value", [
    ("quantized_lm_head", True), ("quantized_embed_tokens", True),
    ("lm_head_untied", True), ("embed_tokens_untied", True),
    ("qaxis", 1), ("qaxis", False), ("mode", "per_group_symmetric"),
    ("default", {"method": "fp16"}), ("group_size", 32),
    ("format_version", 1), ("storage", "int8"),
    ("skip_modules", ["lm_head", "mlp"]),
])
def test_conflicting_legacy_metadata_never_falls_through_to_v1(field, value):
    quant = dict(LEGACY, **{field: value})
    assert not qs.is_legacy_w8a16_config(quant)
    with pytest.raises(UnsupportedModelError, match="legacy W8A16 metadata"):
        loader.parse_quant_config(config(quant))


def test_partial_legacy_metadata_is_rejected_but_modern_v1_remains_uniform():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    partial = dict(LEGACY)
    del partial["quantized_lm_head"]
    with pytest.raises(UnsupportedModelError, match="incomplete or conflicting"):
        loader.parse_quant_config(config(partial))
    modern = {"method": "w8a16"}
    incomplete = config(modern)
    with pytest.raises(UnsupportedModelError, match="exact dense semantics"):
        loader.parse_quant_config(incomplete)
    # Uniform v1 remains valid for the complete official config, not the
    # deliberately minimal fixture used to recognize the old text bundle.
    official = Qwen3_5Config(
        text_config=vars(incomplete.text_config),
        vision_config={"out_hidden_size": 5120}, quant_config=modern,
    )
    assert loader.parse_quant_config(official) == modern
    assert {spec.method for spec in qs.resolve_quant_specs([0, 1], modern).values()} == {qs.W8A16}


@pytest.mark.parametrize("field,value", [
    ("default", {"method": "w8a16"}), ("projection_overrides", {}),
    ("storage", "float16"), ("format_version", 1),
])
def test_reserved_profile_name_cannot_bypass_actual_policy(field, value):
    quant = qs.legacy_w8a16_quant_config()
    quant[field] = value
    with pytest.raises(UnsupportedModelError, match="exact .* policy"):
        loader.parse_quant_config(config(quant))


@pytest.mark.parametrize("field,value", [
    ("hidden_size", 4096), ("num_hidden_layers", 32), ("num_attention_heads", 16),
    ("head_dim", 128), ("linear_num_value_heads", 16),
    ("layer_types", list(reversed(LAYER_TYPES))),
])
def test_legacy_metadata_cannot_route_other_profiles_into_text_only_loader(
    tmp_path, text_loader, field, value
):
    cfg = config()
    setattr(cfg.text_config, field, value)
    with pytest.raises(UnsupportedModelError):
        loader.load_qwen3_5_model(tmp_path, config=cfg)
    assert not text_loader


def test_converter_rejects_legacy_profile_for_other_model_geometry(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    cfg = config()
    cfg.text_config.hidden_size = 4096
    (src / "config.json").write_text(json.dumps({"text_config": vars(cfg.text_config)}))
    with pytest.raises(ValueError, match="exact 27B geometry"):
        convert_checkpoint(src, tmp_path / "converted", profile=qs.LEGACY_W8A16_PROFILE)
    assert not (tmp_path / "converted").exists()


def test_named_legacy_converter_profile_loads_through_public_path(tmp_path, tensors, text_loader):
    src, dst = tmp_path / "source", tmp_path / "converted"
    src.mkdir()
    cfg = config()
    (src / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": vars(cfg.text_config),
    }))
    save_file({key: value.half() for key, value in tensors.items()
               if not key.endswith("_scale")}, src / "model.safetensors")
    convert_checkpoint(src, dst, profile=qs.LEGACY_W8A16_PROFILE)
    emitted = json.loads((dst / "config.json").read_text())
    cfg.quant_config = emitted["quant_config"]
    assert qs.validate_legacy_text_profile(emitted)
    loaded = loader.load_qwen3_5_model(dst, config=cfg)
    assert loaded.model.layers[0].mlp.gate_proj.weight.dtype == torch.int8
    assert loaded.model.layers[0].linear_attn.in_proj_qkv.weight.dtype == torch.float16
    assert not hasattr(loaded.model, "visual")
    assert all(not p.requires_grad for p in loaded.parameters())


@pytest.mark.parametrize("field,value", [
    ("tie_word_embeddings", True), ("hidden_act", "gelu"),
    ("attention_bias", True), ("attn_output_gate", False),
    ("output_gate_type", "sigmoid"), ("rms_norm_eps", 1e-5),
])
def test_same_geometry_cannot_change_hf_loading_or_native_semantics(
    tmp_path, text_loader, field, value
):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    cfg = config()
    text = dict(vars(cfg.text_config), **{field: value})
    # Real HF config normalization preserves these shape-invisible changes.
    hf_config = Qwen3_5Config(text_config=text, quant_config=cfg.quant_config)
    with pytest.raises(UnsupportedModelError, match="original 27B semantics"):
        loader.load_qwen3_5_model(tmp_path, config=hf_config)
    assert text_loader == []  # Reject before from_config/load/tie_weights.


@pytest.mark.parametrize("field,value", [
    ("rope_type", "dynamic"), ("rope_theta", 10000),
    ("partial_rotary_factor", 1.0), ("mrope_interleaved", False),
    ("mrope_section", [10, 11, 11]),
])
def test_rope_semantics_cannot_drift_under_the_legacy_profile(tmp_path, text_loader, field, value):
    cfg = config()
    cfg.text_config.rope_parameters[field] = value
    with pytest.raises(UnsupportedModelError, match="original 27B semantics"):
        loader.load_qwen3_5_model(tmp_path, config=cfg)
    assert not text_loader


def test_nested_hf_defaults_are_equivalent_but_top_level_weight_tying_is_not():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    cfg = config()
    hf_config = Qwen3_5Config(text_config=vars(cfg.text_config), quant_config=cfg.quant_config)
    assert qs.validate_legacy_text_profile(hf_config)
    hf_config.tie_word_embeddings = True
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        qs.validate_legacy_text_profile(hf_config)
