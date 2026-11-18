"""QuantTrio source import and cold admission, without hardware/model forwards."""
from copy import deepcopy
import json
from types import MethodType, SimpleNamespace as NS

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5Config

from test_qwen35_core_profiles import graph_factories, text_install_boundary

from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter, cores, loader, text
from rpu_backend.adapters.qwen3_5 import _rpu_qwen3_5_forward
from rpu_backend.adapters.qwen3_5.quant_scope import (
    FP16, W4A16_AWQ, quant_relpaths, resolve_quant_config, resolve_quant_specs,
    is_full_from_layer_types, validate_declared_dtype,
)
from rpu_backend.api.errors import RPUBackendError
from rpu_backend.quant.awq_gemm import (
    unpack_awq_words, pack_awq_words, unpack_awq_gemm,
    swizzle_pack_uint4_pgrp, swizzle_awq_negative_zero,
)


def config_dict(size=2):
    h, inter, layers, q, kv, vh = {
        2: (2048, 6144, 24, 8, 2, 16), 4: (2560, 9216, 32, 16, 4, 32),
        9: (4096, 12288, 32, 16, 4, 32)}[size]
    return dict(model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"],
        tie_word_embeddings=size != 9, image_token_id=248056,
        quantization_config=dict(quant_method="awq", bits=4, group_size=128,
            version="gemm", zero_point=True,
            modules_to_not_convert=["visual", "linear_attn", "self_attn", "model.layers.0.", "mtp"]),
        text_config=dict(model_type="qwen3_5_text", hidden_size=h, intermediate_size=inter,
            num_hidden_layers=layers, num_attention_heads=q, num_key_value_heads=kv, head_dim=256,
            linear_num_key_heads=16, linear_num_value_heads=vh, linear_key_head_dim=128,
            linear_value_head_dim=128, linear_conv_kernel_dim=4, vocab_size=248320,
            hidden_act="silu", rms_norm_eps=1e-6, attention_bias=False, attn_output_gate=True,
            use_cache=True, full_attention_interval=4,
            layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(layers)],
            rope_parameters=dict(rope_type="default", rope_theta=10_000_000,
                partial_rotary_factor=0.25, mrope_interleaved=True, mrope_section=[11, 11, 10])),
        vision_config=dict(depth=27 if size == 9 else 24, num_heads=16,
            hidden_size=1152 if size == 9 else 1024, intermediate_size=4304 if size == 9 else 4096,
            out_hidden_size=h, hidden_act="gelu_pytorch_tanh", in_channels=3,
            patch_size=16, temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=2304))


@pytest.mark.parametrize("size,count", [(2, 69), (4, 93), (9, 93)])
def test_profile_resolves_raw_and_hf_without_mutation(size, count):
    raw = config_dict(size)
    before = deepcopy(raw)
    quant = resolve_quant_config(raw)
    assert raw == before and "quant_config" not in raw
    config = Qwen3_5Config.from_dict(raw)
    assert resolve_quant_config(config) == quant
    specs = resolve_quant_specs(is_full_from_layer_types(raw["text_config"]["layer_types"]), quant)
    expected = {f"layers.{i}.mlp.{role}_proj" for i in range(1, raw["text_config"]["num_hidden_layers"])
                for role in ("gate", "up", "down")}
    assert {k for k, v in specs.items() if v.method == W4A16_AWQ} == expected
    assert len(expected) == count
    assert all(v.method == FP16 for k, v in specs.items() if k not in expected)
    assert validate_declared_dtype(raw["text_config"]["layer_types"], quant, "w4a16")["profile"] == "quanttrio_awq_g128"


@pytest.mark.parametrize("mutate", [
    lambda c: c["quantization_config"].update(bits=True),
    lambda c: c["quantization_config"].update(group_size=32),
    lambda c: c["quantization_config"].update(zero_point=False),
    lambda c: c["quantization_config"].update(version="gemv"),
    lambda c: c["quantization_config"].update(modules_to_not_convert=["visual"]),
    lambda c: c["text_config"].update(hidden_size=1024),
    lambda c: c["text_config"].update(attn_output_gate=False),
    lambda c: c["text_config"]["layer_types"].__setitem__(0, "full_attention"),
    lambda c: c.update(tie_word_embeddings=False),
    lambda c: c.update(quant_config={"method": "w4a16_pgrp", "group_size": 32}),
])
def test_claimed_awq_neighbours_fail_closed(mutate):
    config = config_dict()
    mutate(config)
    with pytest.raises(ValueError):
        resolve_quant_config(config)


def test_v3_scope_cannot_hide_first_layer_or_attention_change():
    config = config_dict()
    quant = resolve_quant_config(config)
    quant["module_overrides"]["layers.0.mlp.gate_proj"] = {"method": W4A16_AWQ, "group_size": 128}
    with pytest.raises(ValueError, match="exact"):
        resolve_quant_specs(is_full_from_layer_types(config["text_config"]["layer_types"]), quant)


def test_awq_words_use_n_axis_reorder_and_direct_zero():
    known = torch.arange(16, dtype=torch.uint8).reshape(1, 16)
    words = torch.tensor([[0x75316420, 0xfdb9eca8 - 2**32]], dtype=torch.int32)
    assert torch.equal(unpack_awq_words(words), known)
    assert torch.equal(pack_awq_words(known), words)
    q = torch.arange(128 * 16).reshape(128, 16).remainder(16).byte()
    z = torch.arange(16).remainder(16).byte().reshape(1, 16)
    scale = torch.linspace(0.25, 2, 16).half().reshape(1, 16)
    codes, zero, scales = unpack_awq_gemm(pack_awq_words(q), pack_awq_words(z), scale, n=16, k=128)
    assert torch.equal(codes, q.T.contiguous()) and torch.equal(zero, z)
    assert scales is scale
    assert torch.equal((codes.float() - zero.T.float()) * scales.T.float(), torch.zeros(16, 128))


@pytest.mark.parametrize("partition", [0, 1])
def test_uint4_payload_and_zero_metadata_match_independent_addresses(partition):
    n, k, cores_count = 128, 1024, 8
    q = ((torch.arange(n)[:, None] * 131 + torch.arange(k)[None, :] * 7) // 13).remainder(16).byte()
    actual = swizzle_pack_uint4_pgrp(q, partition).numpy()
    expected = []
    local_n, local_k = (n // 8, k) if partition else (n, k // 8)
    for nt in range((local_n + 31) // 32):
        for kt in range(local_k // 32):
            for core in range(cores_count):
                for ni in range(16):
                    for half_n in (0, 16):
                        nl = nt * 32 + half_n + ni
                        ni_global = nl + (core * local_n if partition else 0)
                        for lane in range(16):
                            ki = kt * 32 + lane + (0 if partition else core * local_k)
                            expected.append(0 if nl >= local_n else
                                int(q[ni_global, ki]) | (int(q[ni_global, ki + 16]) << 4))
    assert np.array_equal(actual, np.array(expected, dtype=np.uint8))
    zero = (torch.arange(k // 128)[:, None] + torch.arange(n)[None, :]).remainder(16).byte()
    z = swizzle_awq_negative_zero(zero, partition)
    local_g = k // 128 if partition else k // 128 // 8
    expected_z = torch.zeros(((local_g + 3) // 4, (local_n + 63) // 64, 8, 4, 64), dtype=torch.float16)
    for core in range(8):
        for gi in range(local_g):
            for ni in range(local_n):
                gg = gi if partition else gi + core * local_g
                nn = ni + core * local_n if partition else ni
                expected_z[gi // 4, ni // 64, core, gi % 4, ni % 64] = -int(zero[gg, nn])
    assert torch.equal(z.flatten(), expected_z.flatten()) and z.shape[0] == 128
    q[0, 0] = 16
    with pytest.raises(ValueError, match=r"\[0,15\]"):
        swizzle_pack_uint4_pgrp(q, partition)


class _TinyModel(torch.nn.Module):
    _keys_to_ignore_on_load_unexpected = [r"^mtp\."]

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = torch.nn.Module()
        inner = self.model.language_model = torch.nn.Module()
        inner.layers = torch.nn.ModuleList(torch.nn.Module() for _ in config.text_config.layer_types)
        for path in quant_relpaths(is_full_from_layer_types(config.text_config.layer_types)):
            layer, role, leaf = path.split(".")[1:]
            owner = inner.layers[int(layer)]
            if not hasattr(owner, role):
                owner.add_module(role, torch.nn.Module())
            getattr(owner, role).add_module(leaf, torch.nn.Linear(128, 128, bias=False))
        inner.norm = torch.nn.LayerNorm(128)
        inner.embed_tokens = torch.nn.Embedding(16, 128)
        self.lm_head = torch.nn.Linear(128, 16, bias=False)

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight


def _checkpoint(tmp_path, monkeypatch, *, size=2, missing_zero=False, mutate=None, config=None):
    config = config if config is not None else Qwen3_5Config.from_dict(config_dict(size))
    quant = resolve_quant_config(config)
    model = _TinyModel(config)
    source = {k: torch.full_like(v, 1.003, dtype=torch.bfloat16) for k, v in model.state_dict().items()}
    source["model.language_model.norm.weight"] = torch.full((128,), 1.003, dtype=torch.float32)
    if config.tie_word_embeddings:
        source.pop("lm_head.weight")
    codes = torch.arange(128 * 128).reshape(128, 128).remainder(16).byte()
    zeros = torch.arange(128).reshape(1, 128).remainder(16).byte()
    for name in quant["module_overrides"]:
        stem = "model.language_model." + name
        source.pop(stem + ".weight")
        source[stem + ".qweight"] = pack_awq_words(codes)
        source[stem + ".qzeros"] = pack_awq_words(zeros)
        source[stem + ".scales"] = torch.full((1, 128), 0.125, dtype=torch.float16)
    if missing_zero:
        source.pop("model.language_model.layers.23.mlp.down_proj.qzeros")
    if mutate is not None:
        mutate(source)
    # Place qweight and its scale/zero in different shards intentionally.
    a = {k: v for k, v in source.items() if k.endswith(".qweight")}
    b = {k: v for k, v in source.items() if k not in a}
    save_file(a, str(tmp_path / "one.safetensors"))
    save_file(b, str(tmp_path / "two.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        k: "one.safetensors" if k in a else "two.safetensors" for k in source}}))
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    def skeleton(cfg):
        assert not hasattr(cfg, "quantization_config")
        return _TinyModel(cfg)
    monkeypatch.setattr(loader.AutoModelForImageTextToText, "from_config", skeleton)
    monkeypatch.setattr(loader.AutoModelForImageTextToText, "from_pretrained",
                        lambda *a, **kw: pytest.fail("HF AWQ handler must not run"))
    return config, source


@pytest.mark.parametrize("profile", ["quanttrio_awq_g128", "activation_awq_g128_v1"])
def test_official_27b_awq_loader_reads_real_shards_and_keeps_recipe_identity(tmp_path, monkeypatch, profile):
    from test_qwen35_dense27_profiles import runtime_config27
    config, _ = _checkpoint(tmp_path, monkeypatch, config=runtime_config27(profile))
    model = loader.load_qwen3_5_model(tmp_path, config=config)
    assert loader.validate_awq_loaded_model(model)
    assert model._qwen3_5_awq_import["profile"] == profile
    assert model._qwen3_5_awq_import["triplet_roundtrip_count"] == 63 * 3
    assert model.config.text_config.hidden_size == 5120
    inner = model.model.language_model
    assert model.lm_head.weight is not inner.embed_tokens.weight
    assert inner.layers[0].mlp.gate_proj.weight.dtype == torch.float16
    assert inner.layers[-1].mlp.down_proj.weight.dtype == torch.uint8
    text._validate_projection_specs(inner,
        is_full_from_layer_types(config.text_config.layer_types), model.config.quant_config)


@pytest.mark.parametrize("recipe", ["w4_mlp_only", "w4_mlp_w8_attn_gdn", "w4_main_w8_gdn", "w8_main_w8_gdn"])
def test_official_27b_named_mixed_loader_preserves_actual_projection_storage(tmp_path, monkeypatch, recipe):
    from test_qwen35_dense27_profiles import runtime_config27
    config = runtime_config27(recipe)
    source_model = _TinyModel(config).half()
    state = dict(source_model.state_dict())
    specs = resolve_quant_specs(is_full_from_layer_types(config.text_config.layer_types), config.quant_config)
    for name, spec in specs.items():
        if spec.method == FP16:
            continue
        stem = "model.language_model." + name
        state[stem + ".weight"] = torch.zeros_like(state[stem + ".weight"], dtype=torch.int8)
        shape = (128,) if spec.method == "w8a16" else (128 // spec.group_size, 128)
        state[stem + ".weight_scale"] = torch.ones(shape, dtype=torch.float16)
    save_file(state, str(tmp_path / "model.safetensors"))
    monkeypatch.setattr(loader.AutoModelForImageTextToText, "from_config", _TinyModel)
    model = loader.load_qwen3_5_model(tmp_path, config=config)
    assert model._qwen3_5_quant_config == config.quant_config
    inner = model.model.language_model
    text._validate_projection_specs(inner,
        is_full_from_layer_types(config.text_config.layer_types), config.quant_config)
    assert model.lm_head.weight is not inner.embed_tokens.weight
    for name, spec in specs.items():
        assert inner.get_submodule(name).weight.dtype == (torch.float16 if spec.method == FP16 else torch.int8)


@pytest.mark.parametrize("inference", [False, True])
def test_original_awq_loader_cross_shard_dtype_ties_and_identity(tmp_path, monkeypatch, inference):
    config, source = _checkpoint(tmp_path, monkeypatch)
    load_file, reads = loader.load_file, []
    def read_once(path):
        reads.append(path)
        return load_file(path)
    monkeypatch.setattr(loader, "load_file", read_once)
    before = (tmp_path / "config.json").read_bytes()
    with torch.inference_mode(inference):
        model = loader.load_qwen3_5_model(tmp_path, config=config)
    assert "quant_config" not in vars(config)
    assert (tmp_path / "config.json").read_bytes() == before
    assert loader.validate_awq_loaded_model(model)
    assert reads == [str(tmp_path / "one.safetensors"), str(tmp_path / "two.safetensors")]
    assert model._qwen3_5_awq_import["completed"] is True
    assert model._qwen3_5_awq_import["triplet_roundtrip_count"] == 69
    inner = model.model.language_model
    assert model.lm_head.weight is inner.embed_tokens.weight
    assert torch.equal(inner.norm.weight, source["model.language_model.norm.weight"].half())
    assert not torch.equal(inner.norm.weight, source["model.language_model.norm.weight"].bfloat16().half())
    assert inner.layers[0].mlp.gate_proj.weight.dtype == torch.float16
    linear = inner.layers[23].mlp.down_proj
    assert linear.weight.dtype == linear.weight_zero.dtype == torch.uint8
    assert linear.weight_scale.dtype == torch.float16
    text._validate_projection_specs(inner, is_full_from_layer_types(config.text_config.layer_types), model.config.quant_config)
    # A stale native binary must fail before ownership, dtype/device changes or
    # swizzling; exercising the public to_rpu boundary needs no native handle.
    adapter = object.__new__(Qwen3_5Adapter)
    adapter.model, adapter._rpu_is_ready = model, False
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", NS(_schemas={}), raising=False)
    inventory = loader._awq_inventory(model)
    with pytest.raises(RuntimeError, match="rebuild"):
        adapter.to_rpu()
    assert loader._awq_inventory(model) == inventory
    assert not getattr(model, "_rpu_swizzle_started", False)
    linear.weight_zero[0, 0] = 15
    with pytest.raises(RPUBackendError, match="original CPU loader owners"):
        loader.validate_awq_loaded_model(model)


def test_missing_zero_rejected_before_payload_install(tmp_path, monkeypatch):
    config, _ = _checkpoint(tmp_path, monkeypatch, missing_zero=True)
    monkeypatch.setattr(_TinyModel, "load_state_dict", lambda *a, **kw: pytest.fail("mutation before inventory validation"))
    with pytest.raises(RPUBackendError, match="missing tensors"):
        loader.load_qwen3_5_model(tmp_path, config=config)


@pytest.mark.parametrize("mutate", [
    lambda s: s.update({"model.language_model.layers.1.mlp.up_proj.scales": torch.ones(1, 128)}),
    lambda s: s.update({"model.language_model.layers.1.mlp.up_proj.qzeros": torch.ones(1, 8, dtype=torch.int32)}),
    lambda s: s.update({"model.language_model.layers.0.mlp.up_proj.qweight": torch.ones(128, 16, dtype=torch.int32)}),
    lambda s: s.update({"mtp.layers.0.mlp.up_proj.qweight": torch.ones(128, 16, dtype=torch.int32)}),
])
def test_invalid_awq_headers_fail_before_payload_read(tmp_path, monkeypatch, mutate):
    config, _ = _checkpoint(tmp_path, monkeypatch, mutate=mutate)
    monkeypatch.setattr(loader, "load_file", lambda *a: pytest.fail("payload read before inventory validation"))
    with pytest.raises(RPUBackendError, match="dtype/shape|out-of-scope"):
        loader.load_qwen3_5_model(tmp_path, config=config)


@pytest.mark.parametrize("bad", [0.0, -0.5, float("inf"), float("nan")])
def test_invalid_source_scale_is_rejected_without_requantization(bad):
    q = torch.zeros(128, 16, dtype=torch.int32)
    z = torch.zeros(1, 16, dtype=torch.int32)
    scales = torch.ones(1, 128, dtype=torch.float16)
    scales[0, 7] = bad
    with pytest.raises(ValueError, match="finite and positive"):
        unpack_awq_gemm(q, z, scales, n=128, k=128)


def test_mixed_awq_install_passes_owned_negative_zero_and_releases_raw(monkeypatch):
    """Run real small tensor preparation up to native set_weights, no device.

    Public exact geometry/scope admission is tested separately; only the policy
    resolution is replaced here so real packing can exercise both mixer types.
    """
    from test_g05_w8a16_adapter import _toy_backbone
    from rpu_backend.runtime import _native_retirement
    from rpu_backend.quant.int4_pgrp_pack import swizzle_int4_pgrp_scale
    inner = _toy_backbone()
    inner.layers.append(deepcopy(inner.layers[0]))
    inner.config.layer_types.append("linear_attention")
    inner.config.num_hidden_layers = 3
    inner.config.intermediate_size = 1024
    for i, layer in enumerate(inner.layers):
        for name, n, k in (("gate_proj", 1024, 256), ("up_proj", 1024, 256), ("down_proj", 256, 1024)):
            linear = torch.nn.Linear(k, n, bias=False, dtype=torch.float16)
            if i:
                linear.weight = torch.nn.Parameter(torch.arange(n * k).reshape(n, k).remainder(16).byte(), requires_grad=False)
                linear.register_buffer("weight_scale", torch.full((k // 128, n), 0.125, dtype=torch.float16))
                linear.register_buffer("weight_zero", torch.arange(n * k // 128).reshape(k // 128, n).remainder(16).byte())
            setattr(layer.mlp, name, linear)
    original = deepcopy(inner)
    policy = resolve_quant_config(config_dict())
    policy["module_overrides"] = {k: v for k, v in policy["module_overrides"].items() if int(k.split(".")[1]) < 3}
    monkeypatch.setattr(text, "resolve_quant_config", lambda config: policy)
    original_to = torch.Tensor.to
    def host_to(value, *args, **kwargs):
        args = tuple("cpu" if isinstance(a, (str, torch.device)) and torch.device(a).type == "rpu" else a for a in args)
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_to(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", host_to)
    monkeypatch.setattr(text, "_refresh_qwen35_text_retirement", lambda _: None)
    monkeypatch.setattr(_native_retirement, "_InstalledNativeResource",
        lambda *a, **kw: NS(keepalive=kw["keepalive"], finalizer=None))
    for name, fn in (("qwen3_5_create", lambda: 23), ("qwen3_5_destroy", lambda *_: None),
                     ("qwen3_5_set_linear_acc32", lambda *_: None)):
        monkeypatch.setattr(torch.ops.rpu, name, fn, raising=False)
    captured = {}
    class ReachedNative(Exception):
        pass
    def install(*args, gate_negative_zero=None, up_negative_zero=None, down_negative_zero=None):
        captured.update(args=args, zero=(gate_negative_zero, up_negative_zero, down_negative_zero))
        raise ReachedNative()
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", install, raising=False)
    pending = NS(handle=None)
    with pytest.raises(ReachedNative):
        text._install_qwen3_5_text_for_rpu_impl(inner, inner.config, max_seq_len=64,
            chunk_size_cap=0, exact_chunk_size=0, padding_budget=0, padding_rows="auto",
            execution_config={}, control_snapshot={}, graph_cache=None, prefill_graph=None,
            cpu_stage_weights=False, pending_state=pending)
    for zi, (wi, si, name, partition) in enumerate(((10, 48, "gate_proj", 1), (11, 49, "up_proj", 1), (12, 50, "down_proj", 0))):
        zeros = captured["zero"][zi]
        assert len(zeros) == 3 and zeros[0].numel() == 0
        assert any(owner is zeros for owner in pending.retirement.keepalive)
        assert captured["args"][wi][0].dtype == torch.float16
        for i in (1, 2):
            source = getattr(original.layers[i].mlp, name)
            assert torch.equal(captured["args"][wi][i].flatten(), swizzle_pack_uint4_pgrp(source.weight, partition))
            assert torch.equal(captured["args"][si][i], swizzle_int4_pgrp_scale(source.weight_scale, 128, partition, 8))
            assert torch.equal(zeros[i], swizzle_awq_negative_zero(source.weight_zero, partition))
            assert zeros[i].dtype == torch.float16 and zeros[i].shape == captured["args"][si][i].shape
            installed = getattr(inner.layers[i].mlp, name)
            assert installed.weight.numel() == installed.weight_scale.numel() == installed.weight_zero.numel() == 0


def test_9b_awq_retained_text_is_separate_from_vl_and_reduced(monkeypatch):
    config = Qwen3_5Config.from_dict(config_dict(9))
    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    Qwen3_5Adapter.preflight_execution(config, {"model": {"num_cores": 8}})
    assert text._public_retained_prefill(config, {"model": {"num_cores": 8}}, public_owner=True)
    assert not text._public_retained_prefill(config, public_owner=True, vision_admitted=True)
    with pytest.raises(ValueError, match="Vision"):
        cores.validate_dense_9b_vl_profile(config)
    for count in (4, 6):
        Qwen3_5Adapter.preflight_execution(config, {"model": {"num_cores": count}})


def test_native_capability_requires_all_three_typed_tail_parameters(monkeypatch):
    signature = "test::weights(Tensor[]? nvfp4_tensor_scales=None, Tensor[]? gate_negative_zero=None, Tensor[]? up_negative_zero=None, Tensor[]? down_negative_zero=None) -> ()"
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", NS(_schemas={"": torch._C.parse_schema(signature)}), raising=False)
    text.require_quanttrio_awq_native()
    signature = signature.replace(", Tensor[]? down_negative_zero=None", "")
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", NS(_schemas={"": torch._C.parse_schema(signature)}))
    with pytest.raises(RuntimeError, match="rebuild"):
        text.require_quanttrio_awq_native()


@pytest.mark.parametrize("kwargs", [{"pixel_values": torch.ones(1)}, {"pixel_values_videos": torch.ones(1)},
                                   {"input_ids": torch.ones(2, 1, dtype=torch.long)}])
def test_awq_request_rejects_before_cache_or_runtime_changes(kwargs):
    owner = NS(config=Qwen3_5Config.from_dict(config_dict()))
    before = dict(vars(owner))
    with pytest.raises(NotImplementedError, match="pure text|batch size 1"):
        _rpu_qwen3_5_forward(owner, **kwargs)
    assert vars(owner) == before


def _complete_awq_runtime(model, request):
    from test_qwen3_5_install_lifecycle import _complete_text_model
    from rpu_backend.adapters.qwen3_5 import _qwen3_5_text_runtime_state

    # Reuse the complete native-retirement double from the lifecycle tests;
    # only raw owner release is simulated, with the real install helper.
    _, state = _complete_text_model(request)
    inner = model.model.language_model
    inner._rpu_qwen3_5 = state
    inner._rpu_qwen3_5_text_install_started = True
    model._rpu_swizzle_started = model._rpu_swizzled = True
    model._lm_head_w_rpu_keepalive = object()
    model.forward = MethodType(_rpu_qwen3_5_forward, model)
    assert _qwen3_5_text_runtime_state(model) is state
    return state


def test_awq_adapter_rewrap_uses_complete_live_runtime_after_source_release(
        tmp_path, monkeypatch, request):
    from rpu_backend.adapters.qwen3_5 import _qwen3_5_text_runtime_state

    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    config, _ = _checkpoint(tmp_path, monkeypatch)
    model = loader.load_qwen3_5_model(tmp_path, config=config)
    first = Qwen3_5Adapter(model)
    assert not first._rpu_is_ready
    assert not Qwen3_5Adapter(model)._rpu_is_ready
    state = _complete_awq_runtime(model, request)
    assert Qwen3_5Adapter(model)._handle == state.handle

    text._drop_linear_data(model.model.language_model.layers[1].mlp.gate_proj)
    assert _qwen3_5_text_runtime_state(model) is state
    with pytest.raises(RPUBackendError, match="original CPU loader owners"):
        loader.validate_awq_loaded_model(model)
    repeated = Qwen3_5Adapter(model)
    assert repeated._rpu_is_ready
    assert repeated._handle == state.handle
    assert repeated._graph_cache is state.graph_cache
    assert repeated._execution_session is first._execution_session
    assert repeated.to_rpu(max_seq_len=1) is model


@pytest.mark.parametrize("fault", ["missing_state", "dead_finalizer", "failed_retirement"])
def test_awq_adapter_rewrap_rejects_incomplete_runtime_even_with_original_owners(
        tmp_path, monkeypatch, request, fault):
    from rpu_backend.adapters.qwen3_5 import _qwen3_5_text_runtime_state

    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    config, _ = _checkpoint(tmp_path, monkeypatch)
    model = loader.load_qwen3_5_model(tmp_path, config=config)
    state = _complete_awq_runtime(model, request)
    if fault == "missing_state":
        del model.model.language_model._rpu_qwen3_5
    elif fault == "dead_finalizer":
        state.handle_finalizer.detach()
    else:
        state.retirement_failed = "injected incomplete retirement"
    assert _qwen3_5_text_runtime_state(model) is None
    assert loader.validate_awq_loaded_model(model)
    with pytest.raises(RPUBackendError, match="installation is incomplete"):
        Qwen3_5Adapter(model)


@pytest.mark.parametrize("size", [2, 4, 9])
def test_awq_public_cold_session_retains_prefill_without_large_model_arenas(
        tmp_path, monkeypatch, graph_factories, text_install_boundary, size):
    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    config, _ = _checkpoint(tmp_path, monkeypatch, size=size)
    model = loader.load_qwen3_5_model(tmp_path, config=config)
    owner = Qwen3_5Adapter(model, rpu_execution={"model": {"num_cores": 8}})
    session = owner._execution_session
    assert session.graph_mode == "RETAINED_CACHE"
    signature = ("test::weights(Tensor[]? gate_negative_zero=None, "
                 "Tensor[]? up_negative_zero=None, Tensor[]? down_negative_zero=None) -> ()")
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights",
        NS(_schemas={"": torch._C.parse_schema(signature)}), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_retained_prefill_graph", lambda *a: None, raising=False)
    inner = model.model.language_model
    text_install_boundary.install_qwen3_5_text_for_rpu(
        inner, model.config, execution_config=session.config,
        _public_prefill_graph_mode=session.graph_mode)
    state = inner._rpu_qwen3_5
    assert state.public_retained_prefill
    policy = state.prefill_graph_cache.runtime_policy
    assert policy is state.graph_cache.runtime_policy
    assert policy.graph_arena_count == 0
    assert not policy.qwen35_dense_27b_sdk_budget
    assert not policy.qwen35_legacy_27b_sdk_budget
    assert [name for name, _, _ in graph_factories] == ["GraphCache", "Graph", "GraphCache"]
    assert graph_factories[-1][1]["max_entries"] == 1
