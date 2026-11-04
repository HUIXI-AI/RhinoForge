from types import SimpleNamespace
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import Qwen3Config, Qwen3ForCausalLM
from rpu_backend.adapters import qwen3
from rpu_backend.api import causal_lm, _execution
from rpu_backend.quant import convert_qwen3_w4a16 as w4
from rpu_backend.runtime import decoder, hw_attrs, weights
from rpu_backend.tests.acceptance.test_causal_lm_adapter_install_lifecycle import _Outer, _prepare_case, _NativeReconfigureOps, _InstallInterrupted
from rpu_backend.tests.acceptance.test_decoder_install_transaction import native_install, _FakeTensor, _finalize

def _config(**changes):
    values = dict(architectures=['Qwen3ForCausalLM'], hidden_size=1024, intermediate_size=3072, num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128, vocab_size=151936, tie_word_embeddings=True)
    values.update(changes)
    return Qwen3Config(**values)

def _model(acc32=False, **config):
    model = _Outer(_config(**config))
    with FakeTensorMode():
        actual = Qwen3ForCausalLM._from_config(model.config, dtype=torch.float16)
    (model.model, model.lm_head) = (actual.model, actual.lm_head)
    model._rpu_execution = {'model': {'num_cores': 8}, 'prefill': {'linear_acc32': acc32}}
    return model

@pytest.mark.parametrize('fault', ['size', 'head_dim', 'layers', 'heads', 'vocab', 'rope', 'w8', 'awq', 'empty_metadata', 'tp4', 'tp6'])
def test_recipe_preflight_rejects_before_full_load(monkeypatch, fault):
    config = _config()
    execution = {'model': {'num_cores': 8}}
    if fault == 'size':
        (config.hidden_size, config.intermediate_size) = (1536, 4096)
    elif fault in {'head_dim', 'layers', 'heads', 'vocab'}:
        field = {'head_dim': 'head_dim', 'layers': 'num_hidden_layers', 'heads': 'num_attention_heads', 'vocab': 'vocab_size'}[fault]
        setattr(config, field, getattr(config, field) + 1)
    elif fault == 'rope':
        config.rope_parameters['rope_type'] = 'linear'
    elif fault in {'w8', 'empty_metadata'}:
        config.quant_config = {'method': 'w8a16'} if fault == 'w8' else {}
    elif fault == 'awq':
        config.quantization_config = {'quant_method': 'awq', 'bits': 4}
    else:
        execution['model']['num_cores'] = int(fault[-1])
    monkeypatch.setattr(causal_lm.AutoConfig, 'from_pretrained', lambda *a, **k: config)
    monkeypatch.setattr(causal_lm.AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: pytest.fail('invalid recipe loaded weights'))
    with pytest.raises((qwen3.UnsupportedModelError, ValueError)):
        causal_lm.RPUModelForCausalLM.from_pretrained('source', quantization='w4a16', rpu_execution=execution)

@pytest.mark.parametrize('recipe', ['int4', 'awq', 'W4A16', {}, True])
def test_unknown_recipe_rejected_before_config_load(monkeypatch, recipe):
    monkeypatch.setattr(causal_lm.AutoConfig, 'from_pretrained', lambda *a, **k: pytest.fail('invalid recipe read config'))
    with pytest.raises(qwen3.UnsupportedModelError, match='quantization must'):
        causal_lm.RPUModelForCausalLM.from_pretrained('source', quantization=recipe)

@pytest.mark.parametrize('acc32', [False, True])
@pytest.mark.parametrize('capacity', [None, 16384, 32768])
def test_public_load_is_cpu_first_and_recipe_is_not_an_hf_kwarg(monkeypatch, acc32, capacity):
    model = _model(acc32)
    monkeypatch.setattr(causal_lm.AutoConfig, 'from_pretrained', lambda *a, **k: model.config)
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return model
    monkeypatch.setattr(causal_lm.AutoModelForCausalLM, 'from_pretrained', load)
    loaded = causal_lm.RPUModelForCausalLM.from_pretrained('source', quantization='w4a16', rpu_execution=model._rpu_execution, max_seq_len=capacity, local_files_only=True)
    assert loaded is model and loaded.to.__rpu_quantization__ == 'w4a16'
    assert getattr(model.model, '_rpu_requested_max_seq_len', None) == capacity
    assert calls == [dict(torch_dtype=torch.float16, device_map='cpu', low_cpu_mem_usage=True, local_files_only=True)]
    assert all((p.dtype == torch.float16 and p.device.type == 'cpu' for p in model.parameters()))
    assert not getattr(model, '_rpu_swizzle_started', False)
    with pytest.raises(qwen3.UnsupportedModelError, match='cold-only'):
        qwen3.Qwen3Adapter(model)

@pytest.mark.parametrize('metadata', [None, {'method': 'w4a16'}])
def test_packed_checkpoint_is_not_implicitly_admitted(metadata):
    model = _model()
    if metadata is not None:
        model.config.quant_config = metadata
    projection = model.model.layers[0].self_attn.q_proj
    with projection.weight.fake_mode:
        projection.weight = torch.nn.Parameter(projection.weight.to(torch.uint8), requires_grad=False)
    with pytest.raises(qwen3.UnsupportedModelError, match='Packed UINT8'):
        qwen3.Qwen3Adapter(model)
    with pytest.raises(qwen3.UnsupportedModelError):
        qwen3.Qwen3Adapter(model, quantization='w4a16')

@pytest.mark.parametrize('fault', ['last_projection', 'head', 'embedding', 'norm', 'existing_scale', 'config', 'extra_linear', 'a9'])
def test_changed_cold_source_fails_before_claim_or_pack(monkeypatch, fault):
    model = _model()
    adapter = qwen3.Qwen3Adapter(model, quantization='w4a16')
    with model.lm_head.weight.fake_mode:
        if fault == 'last_projection':
            module = model.model.layers[-1].mlp.down_proj
            module.weight = torch.nn.Parameter(module.weight[:, :-1])
        elif fault in {'head', 'embedding', 'norm'}:
            module = {'head': model.lm_head, 'embedding': model.model.embed_tokens, 'norm': model.model.norm}[fault]
            module.weight = torch.nn.Parameter(module.weight.float())
        elif fault == 'existing_scale':
            model.model.layers[-1].mlp.down_proj.register_buffer('weight_scale', torch.ones(32, 32))
        elif fault == 'config':
            model.config.quant_config = {'method': 'w4a16'}
        elif fault == 'extra_linear':
            model.extra = torch.nn.Linear(16, 16, dtype=torch.float16)
        else:
            model._rpu_unknown_recipe_override = True
    monkeypatch.setattr(qwen3, '_claim_live_instance', lambda *a: pytest.fail('invalid source claimed device'))
    with pytest.raises((qwen3.RPUBackendError, ValueError)):
        adapter.to_rpu()
    assert not getattr(model, '_rpu_swizzle_started', False)

def _host_install(monkeypatch, model, *, error=None):
    (pre, post, stamp) = (hw_attrs.validate_preinstall, hw_attrs.validate_postinstall, hw_attrs.install_hw_attr_validator)
    _prepare_case(monkeypatch, qwen3, model, postinstall_error=error)
    monkeypatch.setattr(hw_attrs, 'validate_preinstall', pre)
    if error is None:
        monkeypatch.setattr(hw_attrs, 'validate_postinstall', post)
    monkeypatch.setattr(hw_attrs, 'install_hw_attr_validator', stamp)
    monkeypatch.setattr(torch.rpu, 'set_caching_allocator', lambda *a: None)
    converted = []

    def swizzle(owner, **kwargs):
        converted.append(kwargs)
        return weights.convert_linear_weights_inplace(owner, **kwargs)
    monkeypatch.setattr(qwen3, 'convert_linear_weights_inplace', swizzle)
    packed = []

    def quantize(weight, group_size):
        packed.append((weight.shape, group_size))
        return (torch.empty_like(weight, dtype=torch.int8), torch.empty(weight.shape[1] // group_size, weight.shape[0], dtype=torch.float16), group_size)
    monkeypatch.setattr(w4, 'quantize_int4_group_wise', quantize)
    monkeypatch.setattr(w4, 'swizzle_pack_int4_pgrp', lambda value, *a: torch.empty(value.numel() // 2, dtype=torch.uint8))
    installs = []
    install = qwen3._install_causal_decoder_forward

    def record(inner, **kwargs):
        installs.append(kwargs)
        return install(inner, **kwargs)
    monkeypatch.setattr(qwen3, '_install_causal_decoder_forward', record)
    return (packed, converted, installs)

@pytest.mark.parametrize('acc32', [False, True])
def test_real_install_keeps_scales_head_accumulation_and_reconfigure_recipe(monkeypatch, acc32):
    model = _model(acc32)
    (packed, converted, installs) = _host_install(monkeypatch, model)
    adapter = qwen3.Qwen3Adapter(model, quantization='w4a16')
    with model.lm_head.weight.fake_mode:
        assert adapter.to_rpu() is model
    assert len(packed) == 28 * 7 and {g for (_, g) in packed} == {32}
    assert converted == [{'skip_names': set(w4.SKIP_LINEAR_NAMES)}]
    assert model.lm_head.weight.dtype == model.model.embed_tokens.weight.dtype == torch.float16
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    assert model.lm_head._rpu_linear_num_cores == 8
    assert installs[0]['execution_config']['prefill']['linear_acc32'] is acc32
    assert installs[0]['topology'].identity() == (2, 8, 8, 8, 8, 8)
    for ((role, _), scales) in zip(w4._PROJECTIONS, installs[0]['scale_lists'], strict=True):
        for (layer, scale) in zip(model.model.layers, scales, strict=True):
            projection = layer.get_submodule(role)
            assert projection.weight.dtype == torch.uint8
            assert projection.weight.shape == (projection.out_features, projection.in_features // 2)
            assert scale is projection.weight_scale and scale.dtype == torch.float16
            assert scale.shape[0] == 32 and scale.ndim == 2
            assert 'weight_scale' in dict(projection.named_buffers())
            assert not hasattr(projection, '_rpu_int4_logical_shape')
    assert decoder._causal_lm_runtime_complete(model)
    hw_attrs.validate_postinstall(model)
    assert qwen3.Qwen3Adapter(model, quantization='w4a16').to_rpu() is model
    with pytest.raises(qwen3.UnsupportedModelError, match='cold-only'):
        qwen3.Qwen3Adapter(model)
    ops = _NativeReconfigureOps([])
    monkeypatch.setattr(decoder, 'torch', SimpleNamespace(ops=SimpleNamespace(rpu=ops)))
    _execution.reconfigure_rpu_execution(model, {'prefill': {'chunk_size': 64}})
    assert ops.current_chunk == 64 and model.to.__rpu_quantization__ == 'w4a16'
    for patch in ({'model': {'num_cores': 6}}, {'prefill': {'linear_acc32': not acc32}}):
        with pytest.raises(ValueError, match='cold-only'):
            _execution.reconfigure_rpu_execution(model, patch)
    assert model.to('rpu') is model and len(packed) == 196 and (len(installs) == 1)

def test_failed_w4_install_retires_graph_and_cannot_pack_again(monkeypatch):
    model = _model()
    (packed, _, _) = _host_install(monkeypatch, model, error=_InstallInterrupted('stop'))
    adapter = qwen3.Qwen3Adapter(model, quantization='w4a16')
    with model.lm_head.weight.fake_mode, pytest.raises(_InstallInterrupted):
        adapter.to_rpu()
    assert 'graph_clear' in model._events and 'handle_finalize' in model._events
    assert not decoder._causal_lm_runtime_complete(model)
    with pytest.raises(qwen3.RPUBackendError, match='prior swizzle'):
        model.to('rpu')
    assert len(packed) == 196

@pytest.mark.parametrize('acc32', [False, True])
@pytest.mark.parametrize('fault', [None, 'skip_layout', 'packed_shape', 'mixed_dtype', 'scale_shape', 'scale_identity', 'scale_dtype', 'nearby_profile', 'missing_scales'])
def test_actual_decoder_installer_admits_only_complete_g32_payload(monkeypatch, native_install, acc32, fault):
    model = _model(acc32)
    _host_install(monkeypatch, model)
    (weights_calls, accumulation) = ([], [])
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_set_weights_w8a16', lambda *args: weights_calls.append(args), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_topology', lambda h: (8, 8, 8, 8, 8), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_set_linear_acc32', lambda h, mode: accumulation.append(mode), raising=False)

    def install(inner, **kwargs):
        projection = inner.layers[-1].mlp.down_proj
        if fault == 'skip_layout':
            projection._rpu_linear_partition = -2
            projection._rpu_linear_num_cores = None
        elif fault == 'packed_shape':
            projection.weight = torch.nn.Parameter(projection.weight[:, :-1].contiguous(), requires_grad=False)
        elif fault == 'mixed_dtype':
            projection.weight = torch.nn.Parameter(projection.weight.to(torch.int8), requires_grad=False)
        elif fault == 'scale_shape':
            projection.weight_scale = projection.weight_scale[:, :-1].contiguous()
            kwargs['scale_lists'][-1][-1] = projection.weight_scale
        elif fault == 'scale_identity':
            kwargs['scale_lists'][-1][-1] = projection.weight_scale.clone()
        elif fault == 'scale_dtype':
            projection.weight_scale = projection.weight_scale.float()
            kwargs['scale_lists'][-1][-1] = projection.weight_scale
        elif fault == 'nearby_profile':
            inner.config.hidden_size = 1536
            inner.config.intermediate_size = 4096
        elif fault == 'missing_scales':
            kwargs['scale_lists'] = None
        table = _FakeTensor()
        return decoder._install_causal_decoder_forward(inner, **kwargs, cos_sin=(table, table))
    monkeypatch.setattr(qwen3, '_install_causal_decoder_forward', install)
    adapter = qwen3.Qwen3Adapter(model, quantization='w4a16')
    with model.lm_head.weight.fake_mode:
        if fault:
            with pytest.raises(ValueError, match='packed Qwen3|weight layout|exact dense'):
                adapter.to_rpu()
            assert native_install['events'] == [] and (not weights_calls)
            return
        assert adapter.to_rpu() is model
    assert len(weights_calls) == 1 and accumulation == [acc32]
    assert native_install['create_topologies'] == [{'num_cores': 8, 'vocab_size': 151936}]
    assert decoder._causal_lm_runtime_complete(model)
    assert weights_calls[0][1][0] is model.model.layers[0].self_attn.q_proj.weight
    assert weights_calls[0][-1][-1] is model.model.layers[-1].mlp.down_proj.weight_scale
    _finalize(model.model)

@pytest.mark.parametrize('partition,n,k', [(1, 128, 64), (0, 16, 512)])
def test_real_pack_matches_existing_g32_physical_payload_without_a9_exceptions(partition, n, k):
    torch.manual_seed(7)
    projection = torch.nn.Linear(k, n, bias=False, dtype=torch.float16)
    (values, scales, _) = w4.quantize_int4_group_wise(projection.weight, 32)
    expected_weight = w4.swizzle_pack_int4_pgrp(values, partition, 8).reshape(n, k // 2)
    expected_scale = w4.swizzle_int4_pgrp_scale(scales, 32, partition, 8)
    w4.pack_linear_int4_(projection, partition, group_size=32)
    assert torch.equal(projection.weight, expected_weight)
    assert torch.equal(projection.weight_scale, expected_scale)
    assert 'weight_scale' in dict(projection.named_buffers())
    assert 'weight_scale' not in projection.state_dict()
    hw_attrs.validate_postinstall(projection)
_DENSE = [(1024, 3072, 28, 16), (2048, 6144, 28, 16), (2560, 9728, 36, 32), (4096, 12288, 36, 32), (5120, 17408, 40, 40), (5120, 25600, 64, 64)]

def _dense_config(shape):
    (h, i, layers, heads) = shape
    return dict(hidden_size=h, intermediate_size=i, num_hidden_layers=layers, num_attention_heads=heads, tie_word_embeddings=h < 4096)

@pytest.mark.parametrize('shape', _DENSE[:5])
@pytest.mark.parametrize('recipe', ['w4a16', 'w4a16_lm_head'])
def test_dense_recipes_cpu_first_load_original_weights(monkeypatch, shape, recipe):
    model = _model(**_dense_config(shape))
    monkeypatch.setattr(causal_lm.AutoConfig, 'from_pretrained', lambda *a, **kw: model.config)
    calls = []

    def load(*a, **kw):
        calls.append(kw)
        return model
    monkeypatch.setattr(causal_lm.AutoModelForCausalLM, 'from_pretrained', load)
    assert causal_lm.RPUModelForCausalLM.from_pretrained('source', quantization=recipe) is model
    assert model.to.__rpu_quantization__ == recipe
    assert 'quantization' not in calls[0] and calls[0]['device_map'] == 'cpu'
    assert model.lm_head.weight.dtype == torch.float16

@pytest.mark.parametrize('recipe', ['w4a16', 'w4a16_lm_head'])
def test_32b_w4_requires_staging_before_hf_weight_load(monkeypatch, recipe):
    config = _config(**_dense_config(_DENSE[-1]))
    monkeypatch.setattr(causal_lm.AutoConfig, 'from_pretrained', lambda *a, **kw: config)
    monkeypatch.setattr(causal_lm.AutoModelForCausalLM, 'from_pretrained', lambda *a, **kw: pytest.fail('materialized 32B FP16 weights'))
    with pytest.raises(qwen3.UnsupportedModelError, match="device='rpu'"):
        causal_lm.RPUModelForCausalLM.from_pretrained('source', quantization=recipe)

@pytest.mark.parametrize('shape', _DENSE[:5])
@pytest.mark.parametrize('acc32', [False, True])
def test_dense_g32_installs_all_layers_without_repacking_head(monkeypatch, shape, acc32):
    model = _model(acc32, **_dense_config(shape))
    (packed, converted, installs) = _host_install(monkeypatch, model)
    adapter = qwen3.Qwen3Adapter(model, quantization='w4a16')
    with model.lm_head.weight.fake_mode:
        adapter.to_rpu()
    assert len(packed) == shape[2] * 7
    assert converted == [{'skip_names': set(w4.SKIP_LINEAR_NAMES)}]
    assert installs[0]['topology'].identity() == (2, 8, 8, 8, 8, 8)
    assert installs[0]['execution_config']['prefill']['linear_acc32'] is acc32
    assert model.lm_head.weight.dtype == model.model.embed_tokens.weight.dtype == torch.float16
    assert adapter.to_rpu() is model

def test_bounded_head_pack_breaks_tie_without_mutating_embedding():
    torch.manual_seed(51)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    model.model.embed_tokens = torch.nn.Embedding(128, 64, dtype=torch.float16)
    model.lm_head = torch.nn.Linear(64, 128, bias=False, dtype=torch.float16)
    model.lm_head.weight = model.model.embed_tokens.weight
    original = model.model.embed_tokens.weight.detach().clone()
    w4.convert_qwen3_to_w4a16_(model, group_size=32, quantize_lm_head=True)
    assert torch.equal(model.model.embed_tokens.weight, original)
    assert model.model.embed_tokens.weight.dtype == torch.float16
    assert model.lm_head.weight.dtype == torch.uint8 and model.lm_head.weight.shape == (128, 32)
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    (expected, scale) = w4.quantize_pack_int4_pgrp_col_bounded(original, 32)
    assert torch.equal(model.lm_head.weight, expected)
    assert torch.equal(model.lm_head.weight_scale, scale)

@pytest.mark.parametrize('acc32', [False, True])
def test_w4_head_real_install_pushes_packed_payload_and_cold_acc(monkeypatch, acc32):
    model = _model(acc32)
    (packed, converted, installs) = _host_install(monkeypatch, model)

    def packed_head(weight, group_size):
        (n, k) = weight.shape
        return (torch.empty(n, k // 2, dtype=torch.uint8), w4.swizzle_int4_pgrp_scale(torch.empty(k // 32, n, dtype=torch.float16), 32, 1, 8))
    monkeypatch.setattr(w4, 'quantize_pack_int4_pgrp_col_bounded', packed_head)
    original_to = torch.Tensor.to

    def to_cpu(tensor, *a, **kw):
        if a and str(a[0]) == 'rpu':
            a = ('cpu', *a[1:])
        return original_to(tensor, *a, **kw)
    monkeypatch.setattr(torch.Tensor, 'to', to_cpu)
    (head_calls, linear_calls) = ([], [])
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_set_lm_head', lambda *a: head_calls.append(a), raising=False)
    monkeypatch.setattr(qwen3, '_text_decode_execution_plan', lambda *a: None)

    def linear(x, weight, bias, partition, cores, accumulation, scale):
        linear_calls.append((weight, partition, cores, accumulation, scale))
        return torch.empty(x.shape[0], model.config.vocab_size, dtype=torch.float16)
    monkeypatch.setattr(torch.ops.rpu, 'linear_with_accumulation', linear, raising=False)
    adapter = qwen3.Qwen3Adapter(model, quantization='w4a16_lm_head')
    with model.lm_head.weight.fake_mode:
        adapter.to_rpu()
        hidden = torch.empty(1, 3, model.config.hidden_size, dtype=torch.float16)
        result = qwen3._int8_lm_head_prefill_logits(model, hidden)
    assert len(packed) == 196 and len(head_calls) == 1
    assert converted == [{'skip_names': set(w4.SKIP_LINEAR_NAMES) | {'lm_head'}}]
    assert head_calls[0][1].dtype == torch.uint8
    assert head_calls[0][2] is model._rpu_lm_head_scale_keepalive
    assert model.model.embed_tokens.weight.dtype == torch.float16
    assert not hasattr(model, '_rpu_lm_head_int8_cpu_ref')
    assert model.to.__rpu_quantization__ == 'w4a16_lm_head'
    assert result.shape == (1, 3, model.config.vocab_size)
    assert linear_calls[0][1:4] == (1, 8, acc32)
    assert linear_calls[0][0] is model._rpu_lm_head_w_keepalive
    assert linear_calls[0][4] is model._rpu_lm_head_scale_keepalive
    assert adapter.to_rpu() is model and len(head_calls) == 1
    with pytest.raises(ValueError, match='cold-only'):
        _execution.reconfigure_rpu_execution(model, {'prefill': {'linear_acc32': not acc32}})
    assert model._rpu_execution['prefill']['linear_acc32'] is acc32

def test_32b_public_staging_claim_arena_banks_and_cache_capacity(monkeypatch):
    from rpu_backend.quant import load_qwen3_quantized as staged
    from rpu_backend.graph import GraphRuntimePolicy
    model = _model(**_dense_config(_DENSE[-1]))
    (packed, converted, installs) = _host_install(monkeypatch, model)
    monkeypatch.setattr(causal_lm, 'require_native_backend', lambda: None)
    events = model._events

    def stage(config, path, **kwargs):
        events.append('stage')
        assert config is model.config and path == 'source'
        assert kwargs == {'quantization': 'w4a16', 'dtype': torch.float16, 'local_files_only': True}
        return model

    def validate(owner, *, quantization):
        assert owner is model and quantization == 'w4a16'
        assert 'claim' not in events
        events.append('validate')
        return True

    def prepare():
        assert 'claim' in events and 'materialize' not in events
        events.append('arena9')
        return True

    def policy(**kwargs):
        assert kwargs == {'graph_arena_count': 9}
        return SimpleNamespace(prepare_arenas=prepare)

    def materialize(owner, *, quantization):
        assert owner is model and quantization == 'w4a16'
        assert owner._rpu_swizzle_started and 'arena9' in events
        events.append('materialize')
        w4.convert_qwen3_to_w4a16_(owner, group_size=32)
        for layer in owner.model.layers:
            for (role, partition) in w4._PROJECTIONS:
                projection = layer.get_submodule(role)
                projection._rpu_linear_partition = partition
                projection._rpu_linear_num_cores = 8
        owner.lm_head._rpu_linear_partition = 1
        owner.lm_head._rpu_linear_num_cores = 8
    monkeypatch.setattr(staged, 'stage_qwen3_w4a16_for_rpu', stage)
    monkeypatch.setattr(staged, 'validate_staged_qwen3_w4a16', validate)
    monkeypatch.setattr(staged, 'materialize_staged_qwen3_w4a16_for_rpu', materialize)
    monkeypatch.setattr(GraphRuntimePolicy, 'from_environment', policy)
    monkeypatch.setattr(causal_lm.AutoConfig, 'from_pretrained', lambda *a, **kw: model.config)
    monkeypatch.setattr(causal_lm.AutoModelForCausalLM, 'from_pretrained', lambda *a, **kw: pytest.fail('full CPU FP16 load'))
    with model.lm_head.weight.fake_mode:
        out = causal_lm.RPUModelForCausalLM.from_pretrained('source', quantization='w4a16', device='rpu', rpu_execution={'prefill': {'chunk_size': 64}}, local_files_only=True)
    assert out is model and (not converted) and (len(packed) == 64 * 7)
    assert events[:5] == ['stage', 'validate', 'validate', 'claim', 'arena9']
    assert events.index('materialize') < events.index('decoder_install')
    assert model._rpu_kv_cache_layer_bank_size == model.model._rpu_kv_cache_layer_bank_size == 8
    assert installs[0]['_graph_cache_max_entries'] == 8
    assert installs[0]['chunk_envelope_for']('qwen3', 64, 5120).bind_capacity(32768) == (32768, 64)
    assert model.to('rpu') is model and events.count('materialize') == 1

def test_32b_w8_banks_decoder_before_remaining_tree(monkeypatch):
    from rpu_backend.quant import load_qwen3_quantized as staged
    from rpu_backend.quant.qwen3_profiles import _W8_METADATA
    from rpu_backend.graph import GraphRuntimePolicy
    model = _model(**_dense_config(_DENSE[-1]))
    model.config.quant_config = dict(_W8_METADATA)
    with model.lm_head.weight.fake_mode:
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight = torch.nn.Parameter(module.weight.to(torch.int8), requires_grad=False)
                module.register_buffer('weight_scale', torch.empty(module.out_features, dtype=torch.float16))
    (packed, converted, installs) = _host_install(monkeypatch, model)
    events = model._events

    def prepare():
        assert events == ['claim']
        events.append('arena9')
        return True
    monkeypatch.setattr(GraphRuntimePolicy, 'from_environment', lambda **kw: SimpleNamespace(prepare_arenas=prepare) if kw == {'graph_arena_count': 9} else pytest.fail('wrong arena budget'))

    def move(owner):
        assert owner is model and owner._rpu_swizzle_started and (events[-1] == 'arena9')
        events.append('decoder_bank')
    monkeypatch.setattr(staged, 'move_qwen3_w8a16_decoder_for_rpu', move)
    monkeypatch.setattr(qwen3, '_collect_embedding_data_ptrs', lambda owner: set())
    monkeypatch.setattr(qwen3, '_apply_fused_lm_head_for_rpu', lambda owner: events.append('head_install'))
    original = qwen3.convert_linear_weights_inplace

    def remaining(owner, **kwargs):
        assert events[-1] == 'decoder_bank'
        assert not any(('layers' in name for (name, _) in owner.named_modules()))
        events.append('remaining')
        return original(owner, **kwargs)
    monkeypatch.setattr(qwen3, 'convert_linear_weights_inplace', remaining)
    adapter = qwen3.Qwen3Adapter(model)
    with model.lm_head.weight.fake_mode:
        adapter.to_rpu()
    assert not packed and len(converted) == 1
    assert events.index('decoder_bank') < events.index('remaining') < events.index('decoder_install')
    assert installs[0]['_graph_cache_max_entries'] == 8
    assert installs[0]['chunk_envelope_for']('qwen3', 64, 5120).bind_capacity(32768) == (32768, 64)
    assert model._rpu_kv_cache_layer_bank_size == model.model._rpu_kv_cache_layer_bank_size == 8
