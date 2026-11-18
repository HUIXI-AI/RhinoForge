"""Local activation-aware conversion, functional folding and loader format."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import Qwen3_5Config

from rpu_backend.adapters.qwen3_5 import loader
from rpu_backend.adapters.qwen3_5 import quant_scope as qs
from rpu_backend.quant import convert_qwen3_5_awq as awq
from rpu_backend.quant.awq_gemm import pack_awq_words, unpack_awq_gemm
from rpu_backend.quant.convert_qwen3_5 import _quant_config
from rpu_backend.tests.quant.test_qwen3_5_awq import _TinyModel, config_dict


def test_full_precision_mlp_scale_folding_keeps_one_centered_rmsnorm_equivalent():
    torch.manual_seed(5)
    x, norm = torch.randn(7, 128), torch.randn(128) / 10
    gate, up, down = torch.randn(256, 128) / 16, torch.randn(256, 128) / 16, torch.randn(128, 256) / 16
    si, sd = torch.exp(torch.randn(128) / 2), torch.exp(torch.randn(256) / 2)
    new_norm, new_gate, new_up, new_down = awq.fold_mlp_scales(norm, gate, up, down, si, sd)
    rms = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    before = (F.silu(F.linear(rms * (1 + norm), gate)) * F.linear(rms * (1 + norm), up)) @ down.t()
    after = (F.silu(F.linear(rms * (1 + new_norm), new_gate))
             * F.linear(rms * (1 + new_norm), new_up)) @ new_down.t()
    torch.testing.assert_close(after, before, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('row_block', [1, 7, 256])
def test_asymmetric_codes_round_trip_in_exact_loader_gemm_format(row_block):
    torch.manual_seed(9)
    weight = torch.randn(16, 256).half()
    weight[0] = 0
    codes, zero, scales = awq.quantize_uint4(weight, row_block=row_block)
    q, z, s = unpack_awq_gemm(pack_awq_words(codes.t().contiguous()),
                             pack_awq_words(zero), scales, n=16, k=256)
    assert torch.equal(q, codes) and torch.equal(z, zero) and torch.equal(s, scales)
    assert bool((scales > 0).all()) and bool(torch.isfinite(scales).all())
    actual = awq.dequantize_uint4(q, z, s)
    assert (actual - weight.float()).abs().max() <= scales.max().float() * 0.51
    assert torch.count_nonzero(actual[0]) == 0


def test_activation_search_uses_real_samples_and_retains_unscaled_candidate():
    torch.manual_seed(11)
    weight, samples = torch.randn(32, 256), torch.randn(12, 256)
    samples[:, :4] *= 80
    scale = awq.search_scale((weight,), samples, grid=5)
    def error(s):
        q, z, wscale = awq.quantize_uint4(weight * s)
        diff = awq.dequantize_uint4(q, z, wscale) / s - weight
        return (samples @ diff.t()).square().mean()
    assert error(scale) <= error(torch.ones(256)) + 1e-7
    assert not torch.equal(scale, torch.ones(256))
    with pytest.raises(ValueError, match='nonempty finite'):
        awq.search_scale((weight,), torch.full_like(samples, float('nan')))


@pytest.mark.parametrize('profile', qs.NAMED_MIXED_PROFILES)
def test_named_official_27b_mixed_recipes_are_exact(profile):
    from test_qwen35_dense27_profiles import config27
    config = config27()
    full = qs.is_full_from_layer_types(config.text_config.layer_types)
    config.quant_config = _quant_config(full, None, 32, profile=profile)
    assert qs.validate_named_27b_mixed_profile(config)
    assert not qs.validate_modern_27b_profile(config, allow_fp16=True)
    config.quant_config['projection_overrides']['mlp.down_proj'] = {'method': 'fp16'}
    with pytest.raises(ValueError, match='exact role policy'):
        qs.validate_named_27b_mixed_profile(config)


def test_local_awq_profile_has_independent_identity_and_rejects_external_provenance():
    raw = config_dict(2)
    raw.pop('quantization_config')
    full = qs.is_full_from_layer_types(raw['text_config']['layer_types'])
    raw['quant_config'] = qs._quanttrio_policy(full, qs.ACTIVATION_AWQ_PROFILE)
    before = deepcopy(raw)
    assert qs.resolve_quant_config(raw)['profile'] == qs.ACTIVATION_AWQ_PROFILE
    assert raw == before
    raw['quantization_config'] = config_dict(2)['quantization_config']
    with pytest.raises(ValueError, match='must not claim'):
        qs.resolve_quant_config(raw)


class TinyWithNorms(_TinyModel):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.language_model.layers:
            layer.post_attention_layernorm = torch.nn.LayerNorm(128, bias=False)


@pytest.fixture
def conversion_source(tmp_path):
    # Tiny tensors exercise export/import storage only; real admitted metadata
    # is retained so the profile parser also runs without a geometry stub.
    src, dst = tmp_path / 'source', tmp_path / 'output'
    src.mkdir()
    raw = config_dict(2)
    raw.pop('quantization_config')
    (src / 'config.json').write_text(json.dumps(raw))
    torch.manual_seed(123)
    model = TinyWithNorms(Qwen3_5Config.from_dict(raw))
    values = {key: tensor.detach().half().contiguous().clone()
              for key, tensor in model.state_dict().items()}
    for key in values:
        if 'post_attention_layernorm.weight' in key:
            values[key].zero_()
    save_file(values, src / 'model.safetensors')
    _, _, _, _, identity = awq.source_inventory(src)
    stats = {f'layers.{i}.{role}': torch.randn(2, 128).half()
             for i in range(1, 24) for role in ('input', 'down')}
    calibration = tmp_path / 'calibration.safetensors'
    save_file(stats, calibration, metadata={
        'format': 'qwen35_awq_calibration_v1', 'source_sha256': identity,
        'prompts_sha256': 'a' * 64, 'tokens_sha256': 'b' * 64})
    return src, dst, calibration


def test_export_streams_new_profile_and_original_loader_accepts_triplets(conversion_source, monkeypatch):
    src, dst, calibration = conversion_source
    original = (src / 'config.json').read_bytes()
    (src / 'SHA256SUMS').write_text('source checkpoint checksum\n')
    awq.convert(src, dst, calibration, grid=2, shard_mb=1)
    assert not (dst / 'SHA256SUMS').exists()
    raw = json.loads((dst / 'config.json').read_text())
    assert raw['quant_config']['profile'] == qs.ACTIVATION_AWQ_PROFILE
    assert 'quantization_config' not in raw
    for owner in (raw, raw['text_config'], raw['vision_config']):
        assert owner['dtype'] == owner['torch_dtype'] == 'float16'
    assert raw['quantization_provenance']['rms_norm_weight'] == 'one_centered'
    mapping = json.loads((dst / 'model.safetensors.index.json').read_text())['weight_map']
    assert len(set(mapping.values())) > 1
    config = Qwen3_5Config.from_dict(raw)
    monkeypatch.setattr(loader.AutoModelForImageTextToText, 'from_config', TinyWithNorms)
    loaded = loader.load_qwen3_5_model(dst, config=config)
    assert loader.validate_awq_loaded_model(loaded)
    assert loaded.model.language_model.layers[1].mlp.gate_proj.weight.dtype == torch.uint8
    assert loaded.model.language_model.layers[0].mlp.gate_proj.weight.dtype == torch.float16
    assert loaded._qwen3_5_awq_import['profile'] == qs.ACTIVATION_AWQ_PROFILE
    assert (src / 'config.json').read_bytes() == original


def test_export_rejects_changed_source_and_cleans_failed_outputs(conversion_source, monkeypatch):
    src, dst, calibration = conversion_source
    def fail(*args, **kwargs):
        raise RuntimeError('injected export failure')
    monkeypatch.setattr(awq, 'pack_awq_words', fail)
    with pytest.raises(RuntimeError, match='injected'):
        awq.convert(src, dst, calibration, grid=2)
    assert not dst.exists() and not list(dst.parent.glob('.output.tmp-*'))
    raw = json.loads((src / 'config.json').read_text())
    raw['irrelevant_change'] = 'still requires recalibration'
    (src / 'config.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='not bound'):
        awq.convert(src, dst, calibration, grid=2)
    assert not dst.exists()


def test_calibration_collects_actual_forward_hooks_and_binds_prompt_tokens(
        conversion_source, tmp_path, monkeypatch):
    from types import MethodType
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    src, _, _ = conversion_source
    model = TinyWithNorms(Qwen3_5Config.from_dict(json.loads((src / 'config.json').read_text())))
    model.load_state_dict(load_file(src / 'model.safetensors'))
    for layer in model.model.language_model.layers:
        def mlp_forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        layer.mlp.forward = MethodType(mlp_forward, layer.mlp)
    def forward(self, input_ids, **_kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = x + layer.mlp(x)
        return x
    model.model.language_model.forward = MethodType(forward, model.model.language_model)
    monkeypatch.setattr(AutoModelForImageTextToText, 'from_pretrained', lambda *a, **k: model)
    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return {'input_ids': torch.tensor([[1, 2, 3, 4]])}
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    prompts = tmp_path / 'prompts.jsonl'
    prompts.write_text('{"text":"Describe a river."}\n"Explain rain."\n')
    output = tmp_path / 'actual_calibration.safetensors'
    assert awq.calibrate(src, prompts, output, max_tokens=8, sample_rows=2) == 2
    with safe_open(output, framework='pt') as samples:
        assert len(list(samples.keys())) == 46
        assert samples.get_tensor('layers.1.input').shape == (4, 128)
        assert samples.metadata()['prompts_sha256'] == awq._digest(prompts)
        assert len(samples.metadata()['tokens_sha256']) == 64
    assert not model.model.language_model.layers[1].mlp._forward_pre_hooks


@pytest.mark.parametrize('fault', ['source_half_overflow', 'folded_norm_half_overflow'])
def test_export_rejects_fp16_overflow_without_publishing(conversion_source, monkeypatch, fault):
    src, dst, calibration = conversion_source
    if fault == 'source_half_overflow':
        values = load_file(src / 'model.safetensors')
        name = 'model.language_model.layers.1.post_attention_layernorm.weight'
        values[name] = torch.full((128,), 100000.0, dtype=torch.float32)
        save_file(values, src / 'model.safetensors')
        _, _, _, _, identity = awq.source_inventory(src)
        with safe_open(calibration, framework='pt') as file:
            metadata = dict(file.metadata())
            samples = {name: file.get_tensor(name) for name in file.keys()}
        metadata['source_sha256'] = identity
        save_file(samples, calibration, metadata=metadata)
    else:
        original = awq.fold_mlp_scales
        def overflow(*args, **kwargs):
            norm, *weights = original(*args, **kwargs)
            return norm.fill_(100000), *weights
        monkeypatch.setattr(awq, 'fold_mlp_scales', overflow)
    with pytest.raises(ValueError, match='overflows FP16'):
        awq.convert(src, dst, calibration, grid=2)
    assert not dst.exists() and not list(dst.parent.glob('.output.tmp-*'))


def test_rtn_conversion_never_copies_weight_backups_as_sidecars(tmp_path):
    from rpu_backend.quant.convert_qwen3_5 import _copy_sidecar_files
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    src.mkdir()
    dst.mkdir()
    keep = {'tokenizer.json', 'tokenizer.model', 'generation_config.json', 'README.md'}
    omit = {'config.json', 'model.safetensors', 'model.safetensors.index.json',
            'old.bin', 'backup.pt', 'training.pth', 'old.safetensors',
            'pytorch_model.bin.index.json', 'old.safetensors.index.json'}
    for name in keep | omit:
        (src / name).write_text('placeholder')
    _copy_sidecar_files(src, dst, {'model.safetensors'})
    assert {path.name for path in dst.iterdir()} == keep


@pytest.mark.parametrize('owner,name', [('root', 'quantization_config'),
                                       ('text', 'quantization_config'), ('text', 'quant_config')])
def test_rtn_conversion_rejects_already_quantized_source_before_output(tmp_path, owner, name):
    from rpu_backend.quant.convert_qwen3_5 import convert_checkpoint
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    src.mkdir()
    value = {'architectures': ['Qwen3_5ForConditionalGeneration'], 'text_config': {}}
    target = value if owner == 'root' else value['text_config']
    target[name] = {'quant_method': 'awq'}
    (src / 'config.json').write_text(json.dumps(value))
    with pytest.raises(ValueError, match='original weights'):
        convert_checkpoint(src, dst, profile='w4_mlp_only')
    assert not dst.exists() and not list(dst.parent.glob('.dst.tmp-*'))


@pytest.mark.parametrize('owner', ['root', 'text'])
@pytest.mark.parametrize('name', ['quant_config', 'quantization_config'])
def test_awq_source_inventory_rejects_quantized_metadata_before_reading_weights(
        tmp_path, owner, name):
    value = {'text_config': {}}
    target = value if owner == 'root' else value['text_config']
    target[name] = {'quant_method': 'awq'}
    (tmp_path / 'config.json').write_text(json.dumps(value))
    # No tensor file is present: refusal must precede calibration or source I/O.
    with pytest.raises(ValueError, match='unquantized original weights'):
        awq.source_inventory(tmp_path)
