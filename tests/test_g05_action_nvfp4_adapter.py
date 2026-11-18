"""Mixed Action precision: cold admission, byte ABI and retained owners."""
from copy import deepcopy
import types

import pytest
import torch

from rpu_backend.adapters.g05 import action, quant
from rpu_backend.adapters.qwen3_5 import text


@pytest.mark.parametrize("mode", ["acc16", "acc32"])
def test_action_precision_is_independent_and_installed_changes_fail(monkeypatch, mode):
    from test_g05_adapter_contracts import _load_adapter
    runtime = _load_adapter()
    assert quant.normalize_action_weight_dtype(None, weight_dtype="w8a16") == "w8a16"
    for value, base in (("nvfp4a16", "fp16"), ("fp16", "w8a16"), (True, "w8a16")):
        with pytest.raises(ValueError):
            quant.normalize_action_weight_dtype(value, weight_dtype=base)
    policy = types.SimpleNamespace(_rpu_g05_policy_ready=True,
        _rpu_g05_weight_dtype="w8a16", _rpu_g05_linear_accumulation=mode,
        _rpu_g05_action_weight_dtype="nvfp4a16")
    monkeypatch.setattr(runtime, "_g05_policy_runtime_ready", lambda _: True)
    assert runtime.patch_g05_policy_for_rpu(policy, weight_dtype="w8a16",
        action_weight_dtype="nvfp4a16", linear_accumulation=mode) is policy
    with pytest.raises(ValueError, match="action_weight_dtype conflicts"):
        runtime.patch_g05_policy_for_rpu(policy, weight_dtype="w8a16", linear_accumulation=mode)
    with pytest.raises(ValueError, match="linear_accumulation conflicts"):
        runtime.patch_g05_policy_for_rpu(policy, weight_dtype="w8a16",
            action_weight_dtype="nvfp4a16", linear_accumulation="acc32" if mode == "acc16" else "acc16")


@pytest.mark.parametrize("mode", ["acc16", "acc32"])
def test_native_schema_checked_before_action_mutation(monkeypatch, mode):
    expert = types.SimpleNamespace()
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", lambda *a: None, raising=False)
    with pytest.raises(RuntimeError, match="nvfp4_tensor_scales"):
        action.patch_g05_action_expert_for_rpu(expert, weight_dtype="w8a16",
            action_weight_dtype="nvfp4a16", linear_accumulation=mode, _g05_exact_k1=True)
    assert vars(expert) == {}
    schema = torch._C.parse_schema("test::set_weights(Tensor[]? nvfp4_tensor_scales=None) -> ()")
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights",
        types.SimpleNamespace(_schemas={"": schema}), raising=False)
    quant.require_g05_action_nvfp4_native()


def test_mixed_acc32_requires_exact_profile_before_mutation():
    expert = types.SimpleNamespace()
    with pytest.raises(ValueError, match="exact K1"):
        action.patch_g05_action_expert_for_rpu(expert, weight_dtype="w8a16",
            action_weight_dtype="nvfp4a16", linear_accumulation="acc32")
    with pytest.raises(TypeError, match="_g05_exact_k1"):
        action.patch_g05_action_expert_for_rpu(expert, weight_dtype="w8a16",
            action_weight_dtype="nvfp4a16", linear_accumulation="acc32", _g05_exact_k1=1)
    assert vars(expert) == {}


def test_nvfp4_split_q_is_packed_from_fp16_with_attention_gate_untouched():
    from rpu_backend.quant.nvfp4_pack import quantize_nvfp4, swizzle_pack_nvfp4_v2
    # Distinct head/query/gate values expose a whole-half rather than per-head split.
    raw = torch.arange(8 * 512, dtype=torch.float32).view(4096, 1).expand(4096, 1024).half().contiguous()
    q, ag = text.split_q_gate(raw, 8, 256)
    assert torch.equal(q[:, 0].view(8, 256), raw.view(8, 512, 1024)[:, :256, 0])
    assert torch.equal(ag[:, 0].view(8, 256), raw.view(8, 512, 1024)[:, 256:, 0])
    codes, scales, tensor_scale = quantize_nvfp4(q)
    packed, striped = swizzle_pack_nvfp4_v2(codes, scales, 1, 8)
    assert packed.dtype == striped.dtype == torch.uint8
    assert tuple(packed.shape) == (2048, 512)
    assert tuple(striped.shape) == (64, 2048)
    assert tensor_scale.dtype == torch.float32
    assert torch.equal(raw[:, 0], torch.arange(4096).half())


@pytest.mark.parametrize("acc32", [False, True])
def test_shared_mixed_install_preserves_byte_scales_and_fp32_table_owners(monkeypatch, acc32):
    from test_g05_w8a16_adapter import _toy_backbone
    from rpu_backend.runtime import _native_retirement
    from rpu_backend.runtime.weights import transform_linear_weight
    inner = _toy_backbone(action=True)
    quant.quantize_g05_text_weights(inner, action=True)
    original = deepcopy(inner)
    # Internal preparation test uses small owners; public admission separately enforces exact K1.
    shapes = ((256, 256), (256, 256), (512, 256), (512, 256), (256, 512))
    payload = {"weights": [[torch.full((n, k // 2), 0xAB, dtype=torch.uint8)] for n, k in shapes],
               "scales": [[torch.full((k // 16, n), 0x38, dtype=torch.uint8)] for n, k in shapes],
               "tensor_scales": [torch.full((24,), 1.000123 + i, dtype=torch.float32) for i in range(5)]}
    original_to = torch.Tensor.to
    def host_to(value, *args, **kwargs):
        args = tuple("cpu" if isinstance(x, str) and x == "rpu" else x for x in args)
        if kwargs.get("device") == "rpu": kwargs["device"] = "cpu"
        return original_to(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", host_to)
    monkeypatch.setattr(text, "_refresh_qwen35_text_retirement", lambda _: None)
    monkeypatch.setattr(_native_retirement, "_InstalledNativeResource",
        lambda *a, **kw: types.SimpleNamespace(keepalive=kw["keepalive"], finalizer=None))
    modes = []
    for name, fn in (("qwen3_5_create", lambda: 23), ("qwen3_5_destroy", lambda *a: None),
                     ("qwen3_5_set_linear_acc32", lambda handle, mode: modes.append((handle, mode)))):
        monkeypatch.setattr(torch.ops.rpu, name, fn, raising=False)
    captured = {}
    class ReachedNative(Exception): pass
    def install(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        raise ReachedNative()
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", install, raising=False)
    state = types.SimpleNamespace(handle=None)
    with pytest.raises(ReachedNative):
        text._install_qwen3_5_text_for_rpu_impl(inner, inner.config, max_seq_len=64,
            chunk_size_cap=0, exact_chunk_size=0, padding_budget=0, padding_rows="auto",
            execution_config={"prefill": {"linear_acc32": acc32}}, control_snapshot={},
            graph_cache=None, prefill_graph=None, cpu_stage_weights=True, pending_state=state,
            g05_w8a16=True, g05_action_nvfp4=payload)
    args = captured["args"]
    assert modes == [(23, acc32)]
    for i, (wi, si) in enumerate(((1, 43), (4, 46), (10, 48), (11, 49), (12, 50))):
        assert torch.equal(args[wi][0], payload["weights"][i][0])
        assert torch.equal(args[si][0], payload["scales"][i][0])
    for actual, expected in zip(captured["kwargs"]["nvfp4_tensor_scales"], payload["tensor_scales"]):
        assert actual.dtype == torch.float32 and torch.equal(actual, expected)
        assert any(owner is actual for owner in state.retirement.keepalive)
    _, gate = text.split_q_gate(original.layers[0].self_attn.q_proj.weight, 8, 32)
    assert torch.equal(args[7][0], transform_linear_weight(gate, 1, 8))
    assert args[2][0].dtype == args[3][0].dtype == args[7][0].dtype == torch.int8
    assert args[44][0].dtype == args[45][0].dtype == args[47][0].dtype == torch.float16


@pytest.mark.parametrize("mode", ["acc16", "acc32"])
def test_action_mixed_failure_keeps_retry_prohibited_and_does_not_publish_precision(monkeypatch, mode):
    from test_g05_w8a16_adapter import _toy_backbone
    expert = _toy_backbone(action=True).eval()
    payload = object()
    monkeypatch.setattr(action, '_check_g05_action_profile', lambda _: None)
    monkeypatch.setattr(quant, 'require_g05_action_nvfp4_native', lambda: None)
    monkeypatch.setattr(quant, 'prepare_g05_action_nvfp4', lambda owner, lanes=8: payload)
    monkeypatch.setattr(quant, 'quantize_g05_text_weights', lambda *a, **kw: None)
    calls = []
    def fail(owner, config, **kwargs):
        assert owner is expert
        assert kwargs['_g05_action_nvfp4'] is payload
        assert kwargs['_g05_w8a16'] is True
        assert kwargs['execution_config'] == {'prefill': {'linear_acc32': mode == 'acc32'}}
        calls.append(kwargs)
        raise RuntimeError('injected native installation failure')
    monkeypatch.setattr(text, 'install_qwen3_5_text_for_rpu', fail)
    kwargs = dict(weight_dtype='w8a16', action_weight_dtype='nvfp4a16',
                  linear_accumulation=mode, _g05_exact_k1=True)
    with pytest.raises(RuntimeError, match='injected native'):
        action.patch_g05_action_expert_for_rpu(expert, **kwargs)
    assert expert._rpu_g05_action_install_started is True
    assert not hasattr(expert, '_rpu_g05_action_ready')
    assert not hasattr(expert, '_rpu_g05_action_weight_dtype')
    with pytest.raises(RuntimeError, match='already attempted'):
        action.patch_g05_action_expert_for_rpu(expert, **kwargs)
    assert len(calls) == 1


def test_effective_inventory_replaces_temporary_w8_projection_rows():
    inventory = [{'name': f'layers.0.{role}', 'method': 'w8a16'} for role in (
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
        'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')]
    payload = {'weights': [[torch.zeros(16, 32, dtype=torch.uint8)] for _ in range(5)],
               'scales': [[torch.zeros(4, 16, dtype=torch.uint8)] for _ in range(5)]}
    result = quant.g05_action_nvfp4_inventory(payload, inventory)
    nv = [item for item in result if item['method'] == 'nvfp4a16']
    assert len(nv) == 5
    assert all(item['tensor_scale_bytes'] == 4 for item in nv)
    assert {item['name'] for item in result if item['method'] == 'w8a16'} == {
        'layers.0.self_attn.q_proj.attention_gate', 'layers.0.self_attn.k_proj', 'layers.0.self_attn.v_proj'}


def test_public_k1_b3_selector_is_cold_exact_and_does_not_silently_fallback():
    from rpu_backend.adapters.g05 import configure_g05_vision_camera_batch
    original = lambda *a, **kw: None
    vision = types.SimpleNamespace(_rpu_vision_k1_b3_available=False,
                                   _rpu_g05_vision_forward_impl=original)
    class Session:
        hot = False
        def require_cold(self):
            if self.hot: raise RuntimeError('already built')
    session = Session()
    policy = types.SimpleNamespace(model=types.SimpleNamespace(vision_tower=vision),
        _execution_session=session, _rpu_g05_fm_only_customer_release=True)
    configure_g05_vision_camera_batch(policy, 1)
    assert vision._rpu_g05_vision_forward_impl is original
    with pytest.raises(ValueError, match='unavailable'):
        configure_g05_vision_camera_batch(policy, 3)
    vision._rpu_vision_k1_b3_available = True
    policy._rpu_g05_fm_only_customer_release = False
    with pytest.raises(ValueError, match='exact G05'):
        configure_g05_vision_camera_batch(policy, 3)
    policy._rpu_g05_fm_only_customer_release = True
    configure_g05_vision_camera_batch(policy, 3)
    wrapped = vision._rpu_g05_vision_forward_impl
    configure_g05_vision_camera_batch(policy, 3)
    assert vision._rpu_g05_vision_forward_impl is wrapped
    session.hot = True
    with pytest.raises(RuntimeError, match='already built'):
        configure_g05_vision_camera_batch(policy, 1)
    assert vision._rpu_g05_vision_forward_impl is wrapped
    session.hot = False
    configure_g05_vision_camera_batch(policy, 1)
    assert vision._rpu_g05_vision_forward_impl is original
