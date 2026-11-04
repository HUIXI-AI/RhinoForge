"""Host contracts for the exact Qwen3.5-35B-A3B TP8 Python integration."""
from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "python/rpu_backend/adapters/qwen3_5_moe"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


QUANT = _load_module("qwen35_moe_quant_scope_contract", ADAPTER / "quant_scope.py")


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_prefill_rope_call_uses_current_cpu_tables_without_upload(monkeypatch, inference, cached):
    """Execute the actual prefill caller and table builder with CPU raw aliases."""
    import ast
    import inspect
    from types import SimpleNamespace

    import torch
    from rpu_backend.adapters.qwen3_5_moe import text

    function = ast.parse(inspect.getsource(text.run_qwen3_5_moe_text)).body[0]
    branch = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "seq_len > 1 and st.mrope_section")
    code = compile(ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])),
                   "actual_moe_prefill_rope", "exec")
    calls = []
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_set_prefill_rope",
                        lambda handle, cos, sin: calls.append((handle, cos, sin)), raising=False)
    original_to = torch.Tensor.to

    def no_upload(tensor, *args, **kwargs):
        if (args and str(args[0]) == "rpu") or str(kwargs.get("device")) == "rpu":
            pytest.fail("MoE prefill table unexpectedly uploaded before its stable setter")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", no_upload)
    with torch.inference_mode(inference):
        state = SimpleNamespace(rotary_dim=8, rope_theta=10000, mrope_section=(1, 1, 2))
        if cached:
            state.prefill_rope_cache = None
        backing = torch.arange(3 * 66).reshape(3, 66)
        positions = backing[:, 1:65].contiguous()
        alias = positions.numpy()
        version = -1 if inference else positions._version
        namespace = dict(vars(text), torch=torch, st=state, handle=71, cache=SimpleNamespace(position=0),
                         seq_len=64, real_len=64, position_ids=positions)
        retained = []
        for offset in (0, 7, 0):
            alias[:] = torch.arange(64).expand(3, -1).numpy() + offset
            assert (-1 if inference else positions._version) == version
            exec(code, namespace)
            handle, cos, sin = calls[-1]
            assert handle == 71 and cos.device.type == sin.device.type == "cpu"
            assert cos.dtype == sin.dtype == torch.float16 and cos.shape == sin.shape == (64, 4)
            assert cos.is_contiguous() and sin.is_contiguous()
            expected = text._build_interleaved_mrope_cos_sin(
                positions, 8, 10000, state.mrope_section, device="cpu")
            assert torch.equal(cos, expected[0]) and torch.equal(sin, expected[1])
            retained.append((cos, sin, cos.clone(), sin.clone()))
            assert all(torch.equal(a, old_a) and torch.equal(b, old_b)
                       for a, b, old_a, old_b in retained)
        assert torch.equal(retained[0][0], retained[2][0])
        assert not torch.equal(retained[0][0], retained[1][0])
        previous = calls[-1][1:]
        exec(code, namespace)
        if cached:
            assert calls[-1][1] is previous[0] and calls[-1][2] is previous[1]
        namespace.update(seq_len=1, real_len=1)
        count = len(calls)
        exec(code, namespace)
        assert len(calls) == count  # Decode retains its static position-lookup tables.


@pytest.mark.parametrize("input_device", ["cpu", "rpu"])
@pytest.mark.parametrize("multimodal", [False, True])
def test_public_forward_reads_current_ids_at_original_embedding_device(
        monkeypatch, input_device, multimodal):
    from types import SimpleNamespace

    import torch
    from rpu_backend.adapters import qwen3_5_moe as moe
    from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache
    from rpu_backend.runtime import hw_attrs

    class RpuIds(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="rpu")

    base = torch.tensor([[1, 2, 3]], dtype=torch.long)
    ids = base if input_device == "cpu" else base.as_subclass(RpuIds)
    version = ids._version
    # Write through a host alias without a Tensor version-counter update.
    alias = base.numpy()
    table = torch.arange(6 * 16, dtype=torch.float16).reshape(6, 16) / 16
    weight = torch.arange(8 * 16, dtype=torch.float16).reshape(8, 16) / 32
    embedding_calls, preflight_calls, fusion_calls, text_calls = [], [], [], []

    def embedding(actual_ids):
        assert actual_ids.device.type == "rpu"
        assert (actual_ids is ids) == (input_device == "rpu")
        assert torch.equal(actual_ids.as_subclass(torch.Tensor), base)
        embedding_calls.append(actual_ids)
        return torch.nn.functional.embedding(actual_ids.as_subclass(torch.Tensor), table)

    inner = SimpleNamespace(embed_tokens=embedding, _rpu_qwen3_5_moe=SimpleNamespace(
        execution_topology=SimpleNamespace(lm_head_tp=8), linear_acc32=False))
    model = SimpleNamespace(model=SimpleNamespace(
        language_model=inner, visual=object(), rope_deltas=None),
        _lm_head_w_rpu_keepalive=weight)
    cache = object.__new__(Qwen3_5MoeCache)
    cache.position = 0
    plan = (64, object())
    bridge = SimpleNamespace(set_mrope_position_delta=lambda _: None)
    position_ids = torch.arange(3).reshape(1, 3)
    pixels, grid = torch.zeros((4, 16)), torch.tensor([[1, 2, 2]])
    token_types = torch.tensor([[0, 1, 0]])

    def preflight(actual_inner, actual_cache, length, *, multimodal_prefill):
        assert actual_inner is inner and actual_cache is cache
        assert length == 3 and multimodal_prefill is multimodal
        preflight_calls.append(length)
        return plan

    def prepare(actual_model, actual_ids, actual_pixels, actual_grid, length, *, _text_bridge):
        assert actual_model is model and actual_ids is ids
        assert actual_pixels is pixels and actual_grid is grid
        assert length == 64 and _text_bridge is bridge
        return False

    def fuse(actual_model, hidden, actual_ids, actual_pixels, actual_grid, **kwargs):
        assert actual_model is model and actual_ids is ids
        assert actual_pixels is pixels and actual_grid is grid
        assert kwargs == {"attention_mask": None, "mm_token_type_ids": token_types,
                          "past_key_values": cache}
        fusion_calls.append(actual_ids)
        return hidden, position_ids

    def run_text(actual_inner, hidden, actual_cache, **kwargs):
        assert actual_inner is inner and actual_cache is cache
        assert hidden.dtype == torch.float16 and hidden.shape == (1, 3, 16)
        assert kwargs["attention_mask"] is None
        assert kwargs["position_ids"] is position_ids
        assert kwargs["multimodal_prefill"] is multimodal
        assert kwargs["preflight_plan"] is plan
        text_calls.append(hidden)
        cache.position += hidden.shape[1]
        return hidden

    monkeypatch.setattr(hw_attrs, "mark_first_forward_done", lambda _: None)
    monkeypatch.setattr(moe, "_preflight_qwen3_5_moe_text", preflight)
    monkeypatch.setattr(moe, "_prepare_qwen3_5_multimodal_spm", prepare)
    monkeypatch.setattr(moe, "make_qwen3_5_moe_vision_text_bridge", lambda _: bridge)
    monkeypatch.setattr(moe, "fuse_visual_embeds", fuse)
    monkeypatch.setattr(moe, "run_qwen3_5_moe_text", run_text)

    def head(flat, actual_weight, bias, partition, cores, acc32):
        assert actual_weight is weight and bias is None
        assert (partition, cores, acc32) == (1, 8, False)
        return torch.nn.functional.linear(flat, weight)

    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", head, raising=False)
    original_to = torch.Tensor.to

    uploads = []

    def upload_ids(tensor, *args, **kwargs):
        if (args and args[0] == "rpu") or kwargs.get("device") == "rpu":
            assert tensor is ids and input_device == "cpu"
            uploads.append(tensor.clone())
            return tensor.clone().as_subclass(RpuIds)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", upload_ids)
    outputs, retained = [], []
    for values in ([[1, 2, 3]], [[3, 1, 4]], [[1, 2, 3]]):
        alias[:] = values
        assert ids._version == version
        cache.position = 0
        image_args = {"pixel_values": pixels, "image_grid_thw": grid,
                      "mm_token_type_ids": token_types} if multimodal else {}
        output = moe._rpu_qwen3_5_moe_forward.__wrapped__(
            model, input_ids=ids, past_key_values=cache,
            attention_mask=torch.ones_like(base), cache_position=torch.arange(3),
            position_ids=position_ids, logits_to_keep=1, **image_args)
        expected = torch.nn.functional.linear(
            torch.nn.functional.embedding(base, table)[:, -1:], weight).float()
        torch.testing.assert_close(output.logits, expected, rtol=0, atol=0)
        assert output.logits.device.type == "cpu" and output.logits.dtype == torch.float32
        assert output.logits.shape == (1, 1, 8) and output.past_key_values is cache
        assert cache.position == 3
        outputs.append(output.logits)
        retained.append(output.logits.clone())
        assert all(torch.equal(old, snapshot) for old, snapshot in zip(outputs, retained))
    assert torch.equal(outputs[0], outputs[2]) and not torch.equal(outputs[0], outputs[1])
    assert len(embedding_calls) == len(preflight_calls) == len(text_calls) == 3
    assert len(fusion_calls) == (3 if multimodal else 0)
    assert len(uploads) == (3 if input_device == "cpu" else 0)
    cache.position = 0
    with pytest.raises(NotImplementedError, match="padded attention"):
        moe._rpu_qwen3_5_moe_forward.__wrapped__(
            model, input_ids=ids, past_key_values=cache, attention_mask=torch.zeros_like(base))
    with pytest.raises(ValueError, match="cache_position"):
        moe._rpu_qwen3_5_moe_forward.__wrapped__(
            model, input_ids=ids, past_key_values=cache, cache_position=torch.arange(1, 4))
    assert len(embedding_calls) == 3 and cache.position == 0


def _exact_config():
    layers = ["linear_attention"] * 3 + ["full_attention"]
    layer_types = layers * 10
    quantized = list(QUANT.expected_fp8_weight_names(layer_types))
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 248320,
            "full_attention_interval": 4,
            "hidden_act": "silu",
            "attention_bias": False,
            "attn_output_gate": True,
            "rms_norm_eps": 1e-6,
            "layer_types": layer_types,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000_000,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
            },
        },
        "vision_config": {
            "model_type": "qwen3_5_moe",
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "out_hidden_size": 2048,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 2304,
            "hidden_act": "gelu_pytorch_tanh",
            "deepstack_visual_indexes": [],
        },
        "quant_config": {
            "method": QUANT.FP8_METHOD,
            "format": QUANT.FP8_FORMAT,
            "weight_mode": 5,
            "qaxis": -1,
            "scale_dtype": "float16",
            "scope": QUANT.FP8_SCOPE,
            "profile": QUANT.FP8_PROFILE,
            "official_block_fp8_compatible": False,
            "quantized_tensors": quantized,
            "quantized_tensor_count": len(quantized),
            "fp16_preserve": list(QUANT.FP16_PRESERVE),
        },
    }


@pytest.mark.parametrize("capacity", [0, -1, True, 512.0])
def test_unsupported_capacity_fails_before_loading_or_allocating(capacity):
    from rpu_backend.adapters.qwen3_5_moe import Qwen3_5MoeAdapter
    from rpu_backend.adapters.qwen3_5_moe.text import install_qwen3_5_moe_text_for_rpu
    from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache

    with pytest.raises(ValueError, match="max_seq_len"):
        Qwen3_5MoeCache.from_config(None, max_seq_len=capacity)
    with pytest.raises(ValueError, match="max_seq_len"):
        Qwen3_5MoeAdapter.to_rpu(None, max_seq_len=capacity)
    with pytest.raises(ValueError, match="max_seq_len"):
        install_qwen3_5_moe_text_for_rpu(None, None, max_seq_len=capacity)


@pytest.mark.parametrize("installed,position", [(512, 512), (16384, 16384)])
def test_decode_cannot_exceed_installed_kv_capacity(monkeypatch, installed, position):
    from types import SimpleNamespace
    from rpu_backend.adapters.qwen3_5.cores import core_topology, topology_tuple
    from rpu_backend.adapters.qwen3_5_moe import text

    topology = core_topology({"model": {"num_cores": 8}})
    state = SimpleNamespace(execution_topology=topology, max_seq_len=installed,
                            chunk_envelope=text._qwen3_5_moe_prefill_envelope(40, 2048, installed))
    owner = SimpleNamespace(_rpu_qwen3_5_moe=state)
    cache = SimpleNamespace(execution_topology=topology_tuple(topology),
                            position=position, max_seq_len=16384)
    monkeypatch.setattr(text, "_qwen35_moe_retirement_ready", lambda value: value is state)
    with pytest.raises(ValueError, match="KV/rotary capacity"):
        text._preflight_qwen3_5_moe_text(owner, cache, 1)


def test_exact_profile_is_full40_mixed_e4m3_and_has_210_typed_tensors():
    config = _exact_config()
    result = QUANT.validate_exact_profile(config)
    assert result["profile"] == QUANT.EXACT_PROFILE_ID
    assert result["quant_config"]["weight_mode"] == 5
    assert result["quant_config"]["qaxis"] == -1
    assert len(result["quant_config"]["quantized_tensors"]) == 210


@pytest.mark.parametrize("mutation", ["debug_layers", "tied_head", "official_fp8"])
def test_near_profiles_fail_closed(mutation):
    config = _exact_config()
    if mutation == "debug_layers":
        config["text_config"]["num_hidden_layers"] = 4
        config["text_config"]["layer_types"] = config["text_config"]["layer_types"][:4]
    elif mutation == "tied_head":
        config["tie_word_embeddings"] = True
    else:
        config["quant_config"]["official_block_fp8_compatible"] = True
    with pytest.raises(ValueError):
        QUANT.validate_exact_profile(config)


def test_text_uses_complete_planner_descriptors_and_has_no_legacy_routes():
    source = (ADAPTER / "text.py").read_text()
    required = {
        "qwen3_5_moe_resolve_prefill_stage_domain",
        "qwen3_5_moe_resolve_decode_stage_descriptor",
        "planned_stage_descriptor",
        "plan_bounded_prefill_execution",
        "plan_native_component_execution",
        "_InstalledNativeResource",
    }
    forbidden = {
        "qwen3_5_moe_get_resolved_chunk_size",
        "qwen3_5_moe_set_chunk_envelope",
        "qwen3_5_moe_set_action_mode",
        "qwen3_5_moe_set_z2_mode",
        "qwen3_5_moe_set_reduced",
        "torch.rpu.get_debug_export",
    }
    assert all(token in source for token in required)
    assert all(token not in source for token in forbidden)


def test_vision_wrapper_directly_imports_the_shared_typed_bridge():
    source = (ADAPTER / "vision.py").read_text()
    assert "from rpu_backend.adapters.qwen3_5.vision import (" in source
    assert "make_qwen3_5_moe_vision_text_bridge" in source
    assert "_lazy_vision_graph_options" in source
    assert "import vision as _shared_vision" not in source
    assert "_mrope_setter" not in source
