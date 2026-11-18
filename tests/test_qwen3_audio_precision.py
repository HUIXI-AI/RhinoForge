"""Speech packing and native-boundary rejection; no device is opened."""
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from transformers import Qwen3Config

from rpu_backend.api import _qwen3_audio_precision as audio
from rpu_backend.quant.int4_pgrp_pack import w4_storage_shape
from rpu_backend.runtime import decoder
from rpu_backend.runtime.topology import resolve_decoder_topology


def _config(width=1024):
    return Qwen3Config(hidden_size=width, intermediate_size=width * 3,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, vocab_size=151936, tie_word_embeddings=True,
        rms_norm_eps=1e-6, max_position_embeddings=65536,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000})


@pytest.mark.parametrize("precision", ["fp16", "w8a16", "w4a16_g32", "nvfp4"])
@pytest.mark.parametrize("partition", [0, 1])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_real_packing_preserves_embedding_and_encodes_nonzero_weights(precision, partition, cores):
    embedding = torch.randn(16 * cores, 64 * cores, generator=torch.Generator().manual_seed(11)).half()
    original = embedding.clone()
    if precision == "nvfp4":
        weight, scale, tensor_scale = audio.pack_audio_nvfp4(embedding, partition, num_cores=cores)
        assert tensor_scale.dtype == torch.float32 and tensor_scale.ndim == 0
        assert torch.isfinite(tensor_scale) and tensor_scale > 0
    else:
        weight, scale = audio.pack_audio_linear(embedding, partition, precision, num_cores=cores)
    assert torch.equal(embedding, original) and embedding.dtype == torch.float16
    assert weight.data_ptr() != embedding.data_ptr() and weight.is_contiguous()
    assert weight.dtype == {"fp16": torch.float16, "w8a16": torch.int8,
                            "w4a16_g32": torch.uint8, "nvfp4": torch.uint8}[precision]
    if precision == "w4a16_g32":
        assert tuple(weight.shape) == w4_storage_shape(*embedding.shape, partition, cores)
    else:
        assert weight.numel() == embedding.numel() // (2 if precision == "nvfp4" else 1)
    assert torch.count_nonzero(weight) > 0
    if precision == "fp16":
        assert scale is None
        # The FP16 physical transform permutes every original element exactly.
        assert torch.equal(weight.flatten().sort().values, original.flatten().sort().values)
    else:
        assert scale is not None and scale.is_contiguous() and torch.count_nonzero(scale) > 0
        assert scale.dtype == (torch.uint8 if precision == "nvfp4" else torch.float16)


def test_nvfp4_cannot_drop_its_tensor_scale_via_two_value_packer():
    with pytest.raises(ValueError, match="NVFP4"):
        audio.pack_audio_linear(torch.ones(128, 512).half(), 1, "nvfp4")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1e10])
@pytest.mark.parametrize("precision", ["fp16", "w8a16", "w4a16_g32", "nvfp4"])
def test_nonfinite_or_fp16_overflow_rejects_before_packing(bad, precision):
    source = torch.ones(128, 512)
    source[0, 0] = bad
    with pytest.raises(ValueError):
        if precision == "nvfp4":
            audio.pack_audio_nvfp4(source, 1)
        else:
            audio.pack_audio_linear(source, 1, precision)


@pytest.mark.parametrize("layers", [5, 28])
def test_tensor_scale_table_preserves_scalars_and_pads_with_identity(layers):
    scalars = [torch.tensor((i + 1) / 32, dtype=torch.float32) for i in range(layers)]
    result = audio.pack_audio_tensor_scales(scalars)
    assert torch.equal(result[:layers], torch.stack(scalars))
    assert result.shape == (((layers + 7) // 8) * 8,)
    assert torch.all(result[layers:] == 1)
    for invalid in (torch.tensor(0.), torch.tensor(float("nan")), torch.tensor([1.]), torch.tensor(1.).half()):
        with pytest.raises(ValueError):
            audio.pack_audio_tensor_scales([invalid, *scalars[1:]])


def _decoder_payload(precision, width=1024, cores=8):
    config = _config(width)
    profile = audio.asr_decoder_profile(config)
    topology = resolve_decoder_topology(num_cores=cores, **profile.geometry())
    layer = torch.nn.Module()
    layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
    roles = (("self_attn.q_proj", 2048, width, 1), ("self_attn.k_proj", 1024, width, 1),
        ("self_attn.v_proj", 1024, width, 1), ("self_attn.o_proj", width, 2048, 0),
        ("mlp.gate_proj", 3 * width, width, 1), ("mlp.up_proj", 3 * width, width, 1),
        ("mlp.down_proj", width, 3 * width, 0))
    scales = []
    for role, n, k, partition in roles:
        linear = torch.nn.Linear(k, n, bias=False, device="meta")
        shape = (n, k) if precision == "w8a16" else (n, k // 2)
        dtype = torch.int8 if precision == "w8a16" else torch.uint8
        linear.weight = torch.nn.Parameter(torch.empty(shape, device="meta", dtype=dtype), requires_grad=False)
        if precision == "w8a16":
            scale_shape, scale_dtype = (n,), torch.float16
        elif precision == "nvfp4":
            scale_shape, scale_dtype = (k // 16, n), torch.uint8
        else:
            component_cores = topology.mlp_tp if role.startswith("mlp.") else topology.attn_tp
            groups, rows = (k // 32, n // component_cores) if partition else (k // 32 // component_cores, n)
            elements = component_cores * ((groups + 3) // 4 * 4) * ((rows + 63) // 64 * 64)
            scale_shape, scale_dtype = (32, elements // 32), torch.float16
        linear.register_buffer("weight_scale", torch.empty(scale_shape, dtype=scale_dtype, device="meta"))
        parent, name = role.split(".")
        setattr(getattr(layer, parent), name, linear)
        scales.append([linear.weight_scale] * 28)
    tables = [torch.empty(32, dtype=torch.float32, device="meta") for _ in range(7)]
    return SimpleNamespace(config=config, layers=[layer] * 28), scales, tables, topology


def _validate(precision, model, scales, tables, topology):
    if precision == "w8a16":
        decoder._validate_decoder_w8_scale_lists(model, scales)
    elif precision == "w4a16_g32":
        decoder._validate_decoder_w4_scale_lists(model, scales, topology, audio_component="asr")
    else:
        decoder._validate_asr_nvfp4_payload(model, scales, tables, topology)


@pytest.mark.parametrize("precision", ["w8a16", "w4a16_g32", "nvfp4"])
@pytest.mark.parametrize("width", [1024, 2048])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_reviewed_decoder_payload_accepts_actual_scale_buffers(precision, width, cores):
    _validate(precision, *_decoder_payload(precision, width, cores))


@pytest.mark.parametrize("precision", ["w8a16", "w4a16_g32", "nvfp4"])
@pytest.mark.parametrize("fault", ["other-owner", "shape", "scale-dtype", "weight-dtype", "missing-layer", "missing-family"])
def test_wrong_scale_payload_rejects_before_native_install(precision, fault):
    model, scales, tables, topology = _decoder_payload(precision)
    linear = model.layers[0].self_attn.q_proj
    if fault == "other-owner":
        scales[0][-1] = scales[0][-1].clone()
    elif fault in ("shape", "scale-dtype"):
        value = linear.weight_scale[:-1] if fault == "shape" else linear.weight_scale.float()
        linear.weight_scale = value
        scales[0] = [value] * 28
    elif fault == "weight-dtype":
        linear.weight = torch.nn.Parameter(linear.weight.half(), requires_grad=False)
    elif fault == "missing-layer":
        scales[0].pop()
    else:
        scales.pop()
    with pytest.raises(ValueError):
        _validate(precision, model, scales, tables, topology)


def test_w4_and_nvfp4_are_not_interchangeable_routes():
    model, scales, tables, topology = _decoder_payload("nvfp4")
    with pytest.raises(ValueError):
        decoder._validate_decoder_w4_scale_lists(model, scales, topology, audio_component="asr")
    model, scales, tables, topology = _decoder_payload("w4a16_g32")
    with pytest.raises(ValueError):
        decoder._validate_asr_nvfp4_payload(model, scales, tables, topology)
    model, scales, tables, topology = _decoder_payload("nvfp4")
    tables[0] = tables[0][:28]
    with pytest.raises(ValueError, match="padded FP32"):
        decoder._validate_asr_nvfp4_payload(model, scales, tables, topology)


def test_asr_installer_rejects_w8_weights_without_scales_before_native_create(monkeypatch):
    model, _, _, topology = _decoder_payload("w8a16")
    layer = model.layers[0]
    layer.self_attn.head_dim = 128
    for owner, name, width in (
        (layer.self_attn, "q_norm", 128), (layer.self_attn, "k_norm", 128),
        (layer, "input_layernorm", 1024), (layer, "post_attention_layernorm", 1024),
    ):
        norm = torch.nn.LayerNorm(width, device="meta", dtype=torch.float16)
        norm.variance_epsilon = 1e-6
        setattr(owner, name, norm)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_create",
        lambda *args, **kwargs: pytest.fail("invalid ASR payload created a native decoder"), raising=False)
    with pytest.raises(ValueError, match="without scales requires complete FP16 projections"):
        decoder._install_causal_decoder_forward(model, arch="qwen3", topology=topology,
            _audio_component="asr", scale_lists=None,
            execution_config={"model": {"num_cores": 8}, "prefill": {"linear_acc32": False}},
            chunk_envelope_for=lambda *args: (512, 128))


def _head_owner(precision, cores=8):
    mode = FakeTensorMode()
    config = _config()
    topology = resolve_decoder_topology(num_cores=cores, **audio.asr_decoder_profile(config).geometry())
    def device_tensor(shape, dtype):
        return FakeTensor(mode, torch.empty(shape, device="meta", dtype=dtype), torch.device("rpu"))
    with torch.device("meta"):
        head = torch.nn.Linear(1024, 151936, bias=False)
    dtype = {"fp16": torch.float16, "w8a16": torch.int8, "w4a16_g32": torch.uint8}[precision]
    storage = w4_storage_shape(151936, 1024, 1, topology.lm_head_tp) if precision == "w4a16_g32" else (151936, 1024)
    head.weight = torch.nn.Parameter(device_tensor(storage, dtype), requires_grad=False)
    if precision != "fp16":
        rows = (151936 // topology.lm_head_tp + 63) // 64 * 64
        shape = (151936,) if precision == "w8a16" else (32, topology.lm_head_tp * rows)
        head.register_buffer("weight_scale", device_tensor(shape, torch.float16))
    head._rpu_linear_num_cores, head._rpu_linear_partition = topology.lm_head_tp, 1
    execution = {"model": {"num_cores": cores}, "prefill": {"linear_acc32": True}}
    model = SimpleNamespace(config=config, _rpu_audio_component="asr", _rpu_decoder_handle=3,
        _rpu_decoder_topology=topology,
        _rpu_execution=execution, _rpu_decoder_retirement_state=SimpleNamespace(keepalive=()))
    return SimpleNamespace(config=config, model=model, lm_head=head, _rpu_execution=execution)


@pytest.mark.parametrize("precision", ["fp16", "w8a16", "w4a16_g32"])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_head_publication_retains_actual_payloads(precision, cores, monkeypatch):
    owner = _head_owner(precision, cores)
    calls = []
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_lm_head", lambda *args: calls.append(args), raising=False)
    audio.push_asr_lm_head(owner, 3, precision)
    assert len(calls) == 1
    assert calls[0][1] is owner._rpu_lm_head_w_keepalive
    assert calls[0][2] is owner._rpu_lm_head_scale_keepalive
    assert owner.model._rpu_decoder_retirement_state.keepalive[-2:] == calls[0][1:]


@pytest.mark.parametrize("fault", ["accumulator", "component", "handle", "core-layout", "nvfp4-head", "architecture", "scale"])
def test_head_rejects_wrong_scope_and_cold_binding_before_native(fault, monkeypatch):
    owner = _head_owner("w8a16")
    precision = "w8a16"
    if fault == "accumulator":
        owner._rpu_execution = {"model": {"num_cores": 8}, "prefill": {"linear_acc32": False}}
    elif fault == "component":
        owner.model._rpu_audio_component = None
    elif fault == "handle":
        owner.model._rpu_decoder_handle = 4
    elif fault == "core-layout":
        owner.lm_head._rpu_linear_num_cores = 4
    elif fault == "nvfp4-head":
        precision = "nvfp4"
    elif fault == "architecture":
        owner.config.architectures = ["Qwen3ForCausalLM"]
    else:
        owner.lm_head.weight_scale = FakeTensor(owner.lm_head.weight_scale.fake_mode,
            torch.empty(151935, dtype=torch.float16, device="meta"), torch.device("rpu"))
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_lm_head", lambda *args: pytest.fail("invalid head published"), raising=False)
    with pytest.raises(ValueError):
        audio.push_asr_lm_head(owner, 3, precision)
