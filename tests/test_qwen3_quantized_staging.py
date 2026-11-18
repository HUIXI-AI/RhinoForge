"""Host loader/resource tests with real safetensors, not model/board evidence."""
import copy
from dataclasses import FrozenInstanceError
import json
import shutil

import pytest
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3Config

from rpu_backend.quant import load_qwen3_quantized as staged
from rpu_backend.quant.convert_qwen3_w4a16 import _PROJECTIONS, pack_linear_int4_, packed_mlp_padded_size
from rpu_backend.quant.int4_pgrp_pack import quantize_pack_int4_pgrp_col_bounded
from rpu_backend.runtime.weights import transform_linear_weight


class TinyStorage(nn.Module):
    """Keep exact config/layer count but reduce physical storage for loader tests."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(128, 512)
        self.model.norm = nn.LayerNorm(128, bias=False)
        self.model.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            layer = nn.Module()
            layer.self_attn, layer.mlp = nn.Module(), nn.Module()
            for role, _ in _PROJECTIONS:
                owner, name = role.split(".")
                getattr(layer, owner).add_module(name, nn.Linear(512, 128, bias=False))
            layer.input_layernorm = nn.LayerNorm(128, bias=False)
            self.model.layers.append(layer)
        self.lm_head = nn.Linear(512, 384, bias=False)


def config():
    return Qwen3Config(
        architectures=["Qwen3ForCausalLM"], hidden_size=5120,
        intermediate_size=25600, num_hidden_layers=64,
        num_attention_heads=64, num_key_value_heads=8, head_dim=128,
        vocab_size=151936, tie_word_embeddings=False, max_position_embeddings=40960,
    )


@pytest.fixture(autouse=True)
def tiny_factory(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(staged.AutoModelForCausalLM, "from_config", TinyStorage)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen3-w4-source")
    cfg = config()
    cfg.save_pretrained(root)
    with torch.device("meta"):
        model = TinyStorage(cfg)
    # BF16 source makes the FP16 conversion exercised rather than assumed.
    data = {name: torch.linspace(-.5, .5, value.numel(), dtype=torch.float32)
            .reshape(value.shape).to(torch.bfloat16)
            for name, value in model.named_parameters()}
    save_file(data, root / "model.safetensors")
    return root, cfg


def stage(source, recipe="w4a16"):
    root, cfg = source
    return staged.stage_qwen3_w4a16_for_rpu(
        copy.deepcopy(cfg), str(root), quantization=recipe, local_files_only=True)


def cpu_device_surrogate(monkeypatch):
    """Exercise packing/bank copies on host; no RPU semantics are claimed."""
    original_empty, original_to = torch.empty, torch.Tensor.to
    allocations = []

    def empty(*args, **kwargs):
        if kwargs.get("device") == "rpu":
            allocations.append((args[0], kwargs.get("dtype")))
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    def to(tensor, *args, **kwargs):
        if args and str(args[0]) in {"rpu", "rpu:0"}:
            args = ("cpu", *args[1:])
        if str(kwargs.get("device")) in {"rpu", "rpu:0"}:
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "to", to)
    # Real release calls are unrelated to bank ownership assertions.
    monkeypatch.setattr("rpu_backend.runtime.weights._release_cpu_weight_pages", lambda: None)
    return allocations


def test_staging_reads_metadata_only_and_plan_is_immutable(source, monkeypatch):
    class MetadataOnly:
        def __init__(self, *args, **kwargs):
            self.handle = safe_open(*args, **kwargs)
        def __enter__(self):
            self.handle.__enter__()
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def keys(self):
            return self.handle.keys()
        def get_slice(self, name):
            return self.handle.get_slice(name)
        def get_tensor(self, name):
            pytest.fail("staging loaded weight payload")

    monkeypatch.setattr(staged, "safe_open", MetadataOnly)
    with torch.inference_mode():
        model = stage(source)
    assert all(value.is_meta for value in model.parameters())
    assert staged.validate_staged_qwen3_w4a16(model, quantization="w4a16")
    with pytest.raises(FrozenInstanceError):
        vars(model)[staged._PLAN_ATTR].quantization = "w4a16_lm_head"
    from rpu_backend.runtime.hw_attrs import validate_preinstall
    validate_preinstall(model)


@pytest.mark.parametrize("fault", ["size", "recipe", "dtype", "config"])
def test_source_admission_fails_before_skeleton_creation(source, monkeypatch, fault):
    cfg, recipe, dtype = copy.deepcopy(source[1]), "w4a16", torch.float16
    if fault == "size":
        cfg.intermediate_size, cfg.num_hidden_layers, cfg.num_attention_heads = 17408, 40, 40
    elif fault == "recipe":
        recipe = "awq"
    elif fault == "dtype":
        dtype = torch.bfloat16
    else:
        cfg.max_position_embeddings += 16
    monkeypatch.setattr(staged.AutoModelForCausalLM, "from_config", lambda *a, **k: pytest.fail("model skeleton"))
    with pytest.raises((ValueError, TypeError), match="32B|requires|differs"):
        staged.stage_qwen3_w4a16_for_rpu(
            cfg, str(source[0]), quantization=recipe, dtype=dtype, local_files_only=True)


@pytest.mark.parametrize("fault", ["recipe", "config", "tensor", "module", "foreign"])
def test_changed_plan_owner_fails_before_allocation(source, monkeypatch, fault):
    model = stage(source)
    recipe = "w4a16"
    if fault == "recipe":
        recipe = "w4a16_lm_head"
    elif fault == "config":
        model.config.max_position_embeddings += 16
    elif fault == "tensor":
        model.lm_head.weight = nn.Parameter(torch.empty_like(model.lm_head.weight))
    elif fault == "module":
        model.model.layers[0].self_attn.q_proj._rpu_linear_partition = 1
    else:
        other = stage(source)
        vars(model)[staged._PLAN_ATTR] = vars(other)[staged._PLAN_ATTR]
    monkeypatch.setattr(staged, "_allocate_decoder_projection_views", lambda *a, **k: pytest.fail("allocation"))
    with pytest.raises(RuntimeError, match="plan|changed"):
        staged.validate_staged_qwen3_w4a16(model, quantization=recipe)


def test_checkpoint_change_after_staging_fails_before_allocation(source, tmp_path, monkeypatch):
    shutil.copy(source[0] / "config.json", tmp_path)
    (tmp_path / "model.safetensors").symlink_to(source[0] / "model.safetensors")
    model = stage((tmp_path, source[1]))
    with (tmp_path / "config.json").open("a") as stream:
        stream.write("\n")
    monkeypatch.setattr(staged, "_allocate_decoder_projection_views", lambda *a, **k: pytest.fail("allocation"))
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        staged.validate_staged_qwen3_w4a16(model, quantization="w4a16")


@pytest.mark.parametrize("fault", ["missing", "dtype", "shape", "unexpected", "index"])
def test_bad_source_metadata_rejects_without_loading_payload(source, tmp_path, fault):
    shutil.copy(source[0] / "config.json", tmp_path)
    name = "lm_head.weight"
    if fault == "index":
        (tmp_path / "model.safetensors").symlink_to(source[0] / "model.safetensors")
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    else:
        shape, dtype = (384, 512), torch.bfloat16
        if fault == "dtype":
            dtype = torch.uint8
        elif fault == "shape":
            shape = (384, 256)
        elif fault == "unexpected":
            name = "unknown.weight"
        save_file({name: torch.zeros(shape, dtype=dtype)}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="metadata|inventory|Unexpected|index"):
        stage((tmp_path, source[1]))


def test_materializer_requires_poison_and_failed_install_is_not_reusable(source, monkeypatch):
    model = stage(source)
    with pytest.raises(RuntimeError, match="ownership/poison"):
        staged.materialize_staged_qwen3_w4a16_for_rpu(model, quantization="w4a16")
    model._rpu_swizzle_started = True
    monkeypatch.setattr(staged, "_move_packed_layer", lambda *a: (_ for _ in ()).throw(RuntimeError("copy failed")))
    with pytest.raises(RuntimeError, match="copy failed"):
        staged.materialize_staged_qwen3_w4a16_for_rpu(model, quantization="w4a16")
    assert vars(model)[staged._PLAN_ATTR] and model._rpu_swizzle_started
    with pytest.raises(RuntimeError, match="inventory changed"):
        staged.validate_staged_qwen3_w4a16(model, quantization="w4a16")


@pytest.mark.parametrize("recipe", ["w4a16", "w4a16_lm_head"])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_real_packing_matches_reference_and_banks_share_only_storage(source, monkeypatch, recipe, cores):
    from rpu_backend.quant.qwen3_profiles import qwen3_w4_topology
    model = stage(source, recipe)
    topology = qwen3_w4_topology(model.config, cores)
    allocations = cpu_device_surrogate(monkeypatch)
    model._rpu_swizzle_started = True
    staged.materialize_staged_qwen3_w4a16_for_rpu(model, quantization=recipe, topology=topology)
    assert staged._PLAN_ATTR not in vars(model)
    assert all(not value.is_meta for value in (*model.parameters(), *model.buffers()))
    assert len(allocations) == 2 * 64
    banks = set()
    with safe_open(source[0] / "model.safetensors", framework="pt") as reader:
        for index, layer in enumerate(model.model.layers):
            weight_owners, scale_owners, pointers = set(), set(), set()
            for role, partition in _PROJECTIONS:
                module = layer.get_submodule(role)
                weight_owners.add(module.weight.untyped_storage().data_ptr())
                scale_owners.add(module.weight_scale.untyped_storage().data_ptr())
                pointers.add(module.weight.data_ptr())
                tp = topology.mlp_tp if role.startswith("mlp.") else topology.attn_tp
                assert (module._rpu_linear_partition, module._rpu_linear_num_cores) == (partition, tp)
                if index == 0:
                    ref = nn.Linear(512, 128, bias=False)
                    ref.weight = nn.Parameter(reader.get_tensor(f"model.layers.{index}.{role}.weight").half())
                    padded = packed_mlp_padded_size(role, ref, topology.mlp_tp)
                    # Six-lane MLP leaves are zero-padded once; others keep their width.
                    assert (padded is not None) == (role.startswith("mlp.") and cores == 6)
                    pack_linear_int4_(ref, partition, group_size=32, num_cores=tp,
                                      padded_size=padded)
                    assert torch.equal(module.weight, ref.weight)
                    assert torch.equal(module.weight_scale, ref.weight_scale)
                    assert (module.out_features, module.in_features) == (
                        ref.out_features, ref.in_features)
            assert len(weight_owners) == len(scale_owners) == 1 and len(pointers) == 7
            assert not banks.intersection(weight_owners)
            banks.update(weight_owners)
        head = reader.get_tensor("lm_head.weight").half()
        if recipe == "w4a16_lm_head":
            weight, scale = quantize_pack_int4_pgrp_col_bounded(head, num_cores=topology.lm_head_tp)
            assert torch.equal(model.lm_head.weight, weight)
            assert torch.equal(model.lm_head.weight_scale, scale)
        else:
            assert torch.equal(model.lm_head.weight, transform_linear_weight(head, 1, num_cores=topology.lm_head_tp))
        assert torch.equal(model.model.embed_tokens.weight,
                           reader.get_tensor("model.embed_tokens.weight").half())
        assert model.model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr()


def test_w8_bank_wrapper_validates_head_before_migration(source, monkeypatch):
    cfg = config()
    from rpu_backend.quant.qwen3_profiles import _W8_METADATA
    cfg.quant_config = copy.deepcopy(_W8_METADATA)
    model = TinyStorage(cfg)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight = nn.Parameter(torch.zeros_like(module.weight, dtype=torch.int8), requires_grad=False)
            module.register_buffer("weight_scale", torch.ones(module.out_features, dtype=torch.float16))
    model.lm_head.weight_scale[0] = 0
    calls = []
    monkeypatch.setattr(staged, "_move_materialized_decoder_for_rpu", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(ValueError, match="positive FP16 scales"):
        staged.move_qwen3_w8a16_decoder_for_rpu(model)
    assert not calls
    model.lm_head.weight_scale[0] = 1
    staged.move_qwen3_w8a16_decoder_for_rpu(model)
    assert calls[0][1] == {"dtype": torch.int8, "per_layer": True,
                           "decoder": model.model, "decoder_prefix": "model"}


def test_plain_w8_banks_reuse_real_swizzle_and_leave_head_for_adapter(monkeypatch):
    cfg = config()
    from rpu_backend.quant.qwen3_profiles import _W8_METADATA
    cfg.quant_config = copy.deepcopy(_W8_METADATA)
    model = TinyStorage(cfg)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            values = (torch.arange(module.weight.numel()) % 127).to(torch.int8).view(module.weight.shape)
            module.weight = nn.Parameter(values, requires_grad=False)
            module.register_buffer("weight_scale", torch.ones(module.out_features, dtype=torch.float16))
    head = model.lm_head.weight
    original = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    allocations = cpu_device_surrogate(monkeypatch)
    model._rpu_swizzle_started = True
    staged.move_qwen3_w8a16_decoder_for_rpu(model)
    assert len(allocations) == 64
    assert model.lm_head.weight is head
    assert torch.equal(model.model.layers[0].self_attn.q_proj.weight,
                       transform_linear_weight(original, 1))
    assert all(len({layer.get_submodule(role).weight.untyped_storage().data_ptr()
                    for role, _ in _PROJECTIONS}) == 1 for layer in model.model.layers)
