import json
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import save_file

from rpu_backend.quant import load as w8a16_load


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(2, 3, bias=False)


class _TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(4, 2)
        self.model.layers = nn.ModuleList([_TinyLayer()])
        self.model.rotary_emb = _TinyRotary()
        self.lm_head = nn.Linear(2, 4, bias=False)


class _TinyRotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(rope_parameters={"rope_type": "default"})
        self.rope_type = "default"
        self.attention_scaling = 0.0
        self.register_buffer("inv_freq", torch.empty(2), persistent=False)
        self.register_buffer("original_inv_freq", torch.empty(2), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config, device=None):
        del config
        return torch.tensor([1.0, 0.5], device=device), 1.0


class _TinyImageText(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(4, 2)
        self.model.language_model.layers = nn.ModuleList([_TinyLayer()])
        self.model.language_model.rotary_emb = _TinyRotary()
        self.model.visual = nn.Module()
        self.model.visual.proj = nn.Linear(2, 2)
        self.lm_head = nn.Linear(2, 4, bias=False)


def _qwen3_vl_32b_w8_config():
    return SimpleNamespace(
        architectures=["Qwen3VLForConditionalGeneration"],
        model_type="qwen3_vl",
        tie_word_embeddings=False,
        quant_config={
            "method": "w8a16",
            "mode": "per_channel_symmetric",
            "qaxis": 0,
            "skip_modules": ["lm_head"],
            "quantized_lm_head": False,
            "quantized_embed_tokens": False,
            "lm_head_untied": False,
            "embed_tokens_untied": False,
        },
        text_config=SimpleNamespace(
            model_type="qwen3_vl_text",
            hidden_size=5120,
            intermediate_size=25600,
            num_hidden_layers=64,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=151936,
            hidden_act="silu",
            rms_norm_eps=1e-6,
        ),
        vision_config=SimpleNamespace(
            model_type="qwen3_vl",
            hidden_size=1152,
            intermediate_size=4304,
            depth=27,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[8, 16, 24],
            out_hidden_size=5120,
            hidden_act="gelu_pytorch_tanh",
        ),
    )


def _tiny_imagetext_checkpoint(**overrides):
    tensors = {
        "model.language_model.embed_tokens.weight": torch.arange(
            8, dtype=torch.float32).reshape(4, 2),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(
            (3, 2), dtype=torch.int8),
        "model.language_model.layers.0.self_attn.q_proj.weight_scale": torch.ones(
            3, dtype=torch.float16),
        "model.visual.proj.weight": torch.ones((2, 2), dtype=torch.float32),
        "model.visual.proj.bias": torch.ones(2, dtype=torch.float32),
        "lm_head.weight": torch.ones((4, 2), dtype=torch.float32),
    }
    tensors.update(overrides)
    return tensors


def test_w8a16_loader_builds_meta_skeleton(monkeypatch, tmp_path):
    seen = {}

    def fake_from_config(config):
        del config
        seen["default_device"] = torch.empty(()).device.type
        return _TinyQwen()

    monkeypatch.setattr(
        w8a16_load.AutoModelForCausalLM, "from_config", fake_from_config)

    save_file(
        {
            "model.embed_tokens.weight": torch.arange(
                8, dtype=torch.float32).reshape(4, 2),
            "lm_head.weight": torch.arange(
                8, dtype=torch.float32).reshape(4, 2),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(
                (3, 2), dtype=torch.int8),
            "model.layers.0.self_attn.q_proj.weight_scale": torch.ones(
                3, dtype=torch.float16),
        },
        str(tmp_path / "model.safetensors"),
    )

    model = w8a16_load.load_w8a16_model(SimpleNamespace(), str(tmp_path))

    assert seen["default_device"] == "meta"
    assert model.model.embed_tokens.weight.dtype == torch.float16
    assert model.lm_head.weight.dtype == torch.float16
    assert model.model.layers[0].self_attn.q_proj.weight.dtype == torch.int8
    assert model.model.layers[0].self_attn.q_proj.weight_scale.dtype == torch.float16
    assert {
        param.device.type for _name, param in model.named_parameters()
    } == {"cpu"}
    assert {
        buf.device.type for _name, buf in model.named_buffers()
    } == {"cpu"}
    assert torch.equal(
        model.model.rotary_emb.inv_freq,
        torch.tensor([1.0, 0.5]),
    )
    assert torch.equal(
        model.model.rotary_emb.original_inv_freq,
        torch.tensor([1.0, 0.5]),
    )


def test_imagetext_loader_is_exact_cpu_only_and_keeps_logical_4304(
    monkeypatch, tmp_path
):
    seen = {}

    def fake_from_config(config):
        seen["config"] = config
        seen["default_device"] = torch.empty(()).device.type
        return _TinyImageText()

    monkeypatch.setattr(
        w8a16_load.AutoModelForImageTextToText, "from_config", fake_from_config
    )
    save_file(_tiny_imagetext_checkpoint(), str(tmp_path / "model.safetensors"))
    config = _qwen3_vl_32b_w8_config()

    model = w8a16_load.load_w8a16_imagetext(config, str(tmp_path))

    assert seen == {"config": config, "default_device": "meta"}
    assert config.vision_config.intermediate_size == 4304
    q_proj = model.model.language_model.layers[0].self_attn.q_proj
    assert q_proj.weight.dtype == torch.int8
    assert q_proj.weight_scale.dtype == torch.float16
    assert tuple(q_proj.weight_scale.shape) == (3,)
    assert model.model.visual.proj.weight.shape == (2, 2)
    assert model.model.visual.proj.weight.dtype == torch.float16
    assert {
        tensor.device.type
        for tensor in (*model.parameters(), *model.buffers())
    } == {"cpu"}


def test_imagetext_rpu_stage_reads_metadata_only(monkeypatch, tmp_path):
    monkeypatch.setattr(
        w8a16_load.AutoModelForImageTextToText,
        "from_config",
        lambda _config: _TinyImageText(),
    )
    save_file(_tiny_imagetext_checkpoint(), str(tmp_path / "model.safetensors"))
    real_safe_open = w8a16_load.safe_open

    class MetadataOnlyHandle:
        def __init__(self, *args, **kwargs):
            self._context = real_safe_open(*args, **kwargs)

        def __enter__(self):
            self._handle = self._context.__enter__()
            return self

        def __exit__(self, *args):
            return self._context.__exit__(*args)

        def keys(self):
            return self._handle.keys()

        def get_slice(self, name):
            return self._handle.get_slice(name)

        def get_tensor(self, name):
            raise AssertionError(f"metadata stage materialized {name}")

    monkeypatch.setattr(w8a16_load, "safe_open", MetadataOnlyHandle)
    model = w8a16_load.stage_w8a16_imagetext_for_rpu(
        _qwen3_vl_32b_w8_config(), str(tmp_path)
    )

    assert all(parameter.is_meta for parameter in model.parameters())
    assert w8a16_load._validate_staged_w8a16_imagetext(model).model_id == id(model)


def test_imagetext_rpu_stage_materializes_nondecoder_then_each_layer(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        w8a16_load.AutoModelForImageTextToText,
        "from_config",
        lambda _config: _TinyImageText(),
    )
    save_file(_tiny_imagetext_checkpoint(), str(tmp_path / "model.safetensors"))
    model = w8a16_load.stage_w8a16_imagetext_for_rpu(
        _qwen3_vl_32b_w8_config(), str(tmp_path)
    )
    plan = w8a16_load._validate_staged_w8a16_imagetext(model)
    events = []
    real_install = w8a16_load._install_imagetext_tensor

    def record_install(*args, **kwargs):
        events.append(("install", args[3]))
        return real_install(*args, **kwargs)

    monkeypatch.setattr(w8a16_load, "_install_imagetext_tensor", record_install)
    monkeypatch.setattr(
        w8a16_load,
        "_swizzle_and_move_w8a16_decoder_layer",
        lambda layer: events.append(("swizzle-move", id(layer))),
    )
    vars(model)["_rpu_swizzle_started"] = True

    w8a16_load._materialize_staged_w8a16_imagetext_for_rpu(model)

    installed = [name for event, name in events if event == "install"]
    decoder_names = set(plan.layer_tensor_names[0])
    assert set(plan.nondecoder_tensor_names).issubset(installed)
    assert decoder_names.issubset(installed)
    assert max(installed.index(name) for name in plan.nondecoder_tensor_names) < min(
        installed.index(name) for name in decoder_names
    )
    assert events[-1] == ("swizzle-move", id(model.model.language_model.layers[0]))
    assert w8a16_load._W8A16_IMAGETEXT_STAGE_ATTR not in vars(model)
    assert not any(parameter.is_meta for parameter in model.parameters())


def test_imagetext_rpu_stage_refuses_tensor_load_before_adapter_poison(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        w8a16_load.AutoModelForImageTextToText,
        "from_config",
        lambda _config: _TinyImageText(),
    )
    save_file(_tiny_imagetext_checkpoint(), str(tmp_path / "model.safetensors"))
    model = w8a16_load.stage_w8a16_imagetext_for_rpu(
        _qwen3_vl_32b_w8_config(), str(tmp_path)
    )
    monkeypatch.setattr(
        w8a16_load,
        "_load_imagetext_plan_names",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tensor load ran before adapter poison")
        ),
    )

    try:
        w8a16_load._materialize_staged_w8a16_imagetext_for_rpu(model)
    except RuntimeError as exc:
        assert "ownership/poison" in str(exc)
    else:
        raise AssertionError("metadata-only stage loaded without adapter poison")


def test_imagetext_metadata_stage_checks_index_shard_mapping(monkeypatch, tmp_path):
    monkeypatch.setattr(
        w8a16_load.AutoModelForImageTextToText,
        "from_config",
        lambda _config: _TinyImageText(),
    )
    tensors = _tiny_imagetext_checkpoint()
    names = sorted(tensors)
    split = len(names) // 2
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    save_file({name: tensors[name] for name in names[:split]}, str(tmp_path / first))
    save_file({name: tensors[name] for name in names[split:]}, str(tmp_path / second))
    weight_map = {
        name: first if name in names[:split] else second for name in names
    }
    weight_map[names[0]] = second
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )

    try:
        w8a16_load.stage_w8a16_imagetext_for_rpu(
            _qwen3_vl_32b_w8_config(), str(tmp_path)
        )
    except KeyError as exc:
        assert "index maps" in str(exc)
    else:
        raise AssertionError("metadata stage accepted a wrong index shard mapping")


def test_imagetext_loader_rejects_nonexact_profile_before_checkpoint_io(tmp_path):
    config = _qwen3_vl_32b_w8_config()
    config.vision_config.intermediate_size = 4352

    try:
        w8a16_load.load_w8a16_imagetext(config, str(tmp_path / "missing"))
    except ValueError as exc:
        assert "restricted to the exact Qwen3-VL" in str(exc)
    else:
        raise AssertionError("non-exact image-text profile reached checkpoint I/O")


@torch.no_grad()
def test_imagetext_loader_rejects_projection_and_scale_contract_violations(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        w8a16_load.AutoModelForImageTextToText,
        "from_config",
        lambda _config: _TinyImageText(),
    )
    config = _qwen3_vl_32b_w8_config()
    weight_name = "model.language_model.layers.0.self_attn.q_proj.weight"
    scale_name = weight_name[: -len(".weight")] + ".weight_scale"

    cases = (
        ({weight_name: torch.ones((3, 2), dtype=torch.float16)}, TypeError, "must be int8"),
        ({scale_name: torch.ones(3, dtype=torch.float32)}, TypeError, "must be fp16"),
        ({scale_name: torch.ones(2, dtype=torch.float16)}, ValueError, "expected (3,)"),
        ({"model.visual.proj.weight": torch.ones((3, 2))}, ValueError, "expected (2, 2)"),
    )
    for index, (overrides, error_type, match) in enumerate(cases):
        case_dir = tmp_path / str(index)
        case_dir.mkdir()
        save_file(
            _tiny_imagetext_checkpoint(**overrides),
            str(case_dir / "model.safetensors"),
        )
        try:
            w8a16_load.load_w8a16_imagetext(config, str(case_dir))
        except error_type as exc:
            assert match in str(exc)
        else:
            raise AssertionError(f"invalid W8A16 case {index} was accepted")


def test_imagetext_loader_requires_complete_scale_inventory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        w8a16_load.AutoModelForImageTextToText,
        "from_config",
        lambda _config: _TinyImageText(),
    )
    tensors = _tiny_imagetext_checkpoint()
    del tensors["model.language_model.layers.0.self_attn.q_proj.weight_scale"]
    save_file(tensors, str(tmp_path / "model.safetensors"))

    try:
        w8a16_load.load_w8a16_imagetext(
            _qwen3_vl_32b_w8_config(), str(tmp_path)
        )
    except KeyError as exc:
        assert "scale inventory mismatch" in str(exc)
    else:
        raise AssertionError("W8A16 image-text loader accepted a missing scale")
