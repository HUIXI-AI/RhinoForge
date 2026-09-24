"""Runtime checkpoint math, admission and CPU ownership; no model/board execution."""
import copy
from dataclasses import replace
import json

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import Qwen3VLConfig

from rpu_backend.quant import convert_qwen3_vl as convert
from rpu_backend.quant import load_qwen3_vl_runtime as runtime
from rpu_backend.quant import load
from rpu_backend.quant.int4_pgrp_pack import quantize_int4_group_wise
from test_qwen3_vl_awq_packing import exact_config, inverse_rpu_weight, inverse_rpu_scale, source_pack


def config(size=2, bits=4, *, source=False):
    result = exact_config(size)
    del result.quantization_config
    result.tie_word_embeddings = source
    result.text_config.tie_word_embeddings = source
    if not source:
        result.quant_config = convert.runtime_quant_config(result.text_config.hidden_size, bits)
    return result


def awq_runtime_config(size=2):
    result = config(size, 4)
    result.quant_config = convert.runtime_quant_config(result.text_config.hidden_size, 4,
        source_quantization=convert.AWQ_SOURCE_QUANTIZATION)
    return result


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('size,bits', [(2,8),(2,4),(4,4)])
def test_exact_new_profiles_and_legacy_formats_stay_separate(size,bits):
    cfg = config(size,bits)
    assert runtime.is_qwen3_vl_runtime_quant_config(cfg)
    assert runtime.is_qwen3_vl_runtime_quant_config(cfg.to_dict())
    assert not load.is_qwen3_vl_2b_w8a16_config(cfg)
    assert not load.is_qwen3_vl_4b_w8a16_config(cfg)
    assert not runtime.is_qwen3_vl_runtime_quant_config(exact_config(size))
    assert not runtime.is_qwen3_vl_runtime_quant_config(config(4,8))
    assert convert.runtime_quant_config(2560,8) == convert.QUANT_CONFIG


@pytest.mark.parametrize('size', [2, 4])
def test_awq_runtime_metadata_is_exact_and_separate_from_rtn_and_original_awq(size):
    cfg = awq_runtime_config(size)
    assert runtime.is_qwen3_vl_runtime_quant_config(cfg)
    assert runtime.is_qwen3_vl_runtime_quant_config(cfg.to_dict())
    assert cfg.quant_config['profile'] == f'qwen3_vl_{size}b_w4a16_awq_runtime_align_v1'
    assert cfg.quant_config['source_quantization'] == 'compressed_tensors_awq'
    assert cfg.quant_config['recipe'] == 'preserve_awq_text7_fp16_rtn_head_w4_vision_w8_v1'
    assert not load.is_qwen3_vl_awq_config(cfg)
    rtn = config(size, 4).quant_config
    assert 'source_quantization' not in rtn and 'recipe' not in rtn
    for field in ('profile', 'source_quantization', 'recipe'):
        changed = copy.deepcopy(cfg)
        changed.quant_config.pop(field)
        assert not runtime.is_qwen3_vl_runtime_quant_config(changed)
        changed.quant_config[field] = 'unregistered'
        assert not runtime.is_qwen3_vl_runtime_quant_config(changed)
    with pytest.raises(ValueError, match='bits=4'):
        convert.runtime_quant_config(cfg.text_config.hidden_size, 8,
            source_quantization=convert.AWQ_SOURCE_QUANTIZATION)


@pytest.mark.parametrize('fault', ['bits','profile','storage','scale_storage','group','group_float',
                                  'extra','head','tie','nested','geometry','float_geometry','rope','vision'])
def test_runtime_admission_rejects_unregistered_scope_and_geometry(fault):
    cfg=config()
    if fault=='bits': cfg.quant_config['method']='w2a16'
    elif fault=='profile': cfg.quant_config['profile']='generic'
    elif fault=='storage': cfg.quant_config['weight_storage']='controller_pgrp'
    elif fault=='scale_storage': cfg.quant_config['scale_storage']='logical_fp16_n_g'
    elif fault=='group': cfg.quant_config['group_size']=64
    elif fault=='group_float': cfg.quant_config['group_size']=32.0
    elif fault=='extra': cfg.quant_config['zero_point']=0
    elif fault=='head': cfg.quant_config['quantized_lm_head']=False
    elif fault=='tie': cfg.tie_word_embeddings=True
    elif fault=='nested': cfg.text_config.quant_config={}
    elif fault=='geometry': cfg.text_config.intermediate_size+=512
    elif fault=='float_geometry': cfg.text_config.num_hidden_layers=28.0
    elif fault=='rope': cfg.text_config.rope_parameters['mrope_section']=[20,20,24]
    else: cfg.vision_config.out_hidden_size=2560
    assert not runtime.is_qwen3_vl_runtime_quant_config(cfg)


@pytest.mark.parametrize('partition,n,k',[(1,128,512),(0,128,9728)])
def test_converter_signed_nibbles_roundtrip_through_actual_controller_pack(partition,n,k):
    weight=((torch.arange(n*k).reshape(n,k)%513)-256).half()/17
    weight[0].zero_()
    packed, logical_scale=convert.quantize_weight(weight,bits=4)
    q,expected_scale,_=quantize_int4_group_wise(weight,32)
    # Source file is ordinary adjacent-K nibble packing, with no AWQ +8 offset.
    decoded=torch.stack((packed&15,packed>>4),-1).reshape(n,k).to(torch.int8)
    decoded=torch.where(decoded>=8,decoded-16,decoded)
    assert torch.equal(decoded,q) and torch.equal(logical_scale,expected_scale)
    device_weight,device_scale=runtime._pack_w4_projection(packed,logical_scale,n=n,k=k,partition=partition)
    assert torch.equal(inverse_rpu_weight(device_weight,n,k,partition),q)
    assert torch.equal(inverse_rpu_scale(device_scale,n,k,partition),logical_scale.t())
    assert torch.equal(q.view(n,k//32,32).half()*logical_scale.t().unsqueeze(-1),
                       decoded.view(n,k//32,32).half()*expected_scale.t().unsqueeze(-1))


@pytest.mark.parametrize('bits',[8,4])
@pytest.mark.parametrize('fault',['weight_dtype','weight_shape','scale_dtype','scale_shape','nan','zero','negative'])
def test_quantized_payload_rejects_invalid_weight_or_scale(bits,fault):
    n,k=128,512
    weight,scale=convert.quantize_weight(torch.ones(n,k),bits=bits)
    if fault=='weight_dtype': weight=weight.half()
    elif fault=='weight_shape': weight=weight[:1]
    elif fault=='scale_dtype': scale=scale.float()
    elif fault=='scale_shape': scale=scale.reshape(-1) if bits==4 else scale.reshape(1,-1)
    else: scale.flatten()[0]={'nan':float('nan'),'zero':0,'negative':-1}[fault]
    with pytest.raises(ValueError):
        if bits==4: runtime._pack_w4_projection(weight,scale,n=n,k=k,partition=1)
        else: runtime._copy_w8_projection(weight,scale,n=n,k=k)


class TinyStorage(nn.Module):
    """Injected storage geometry only; real config/admission and layer count remain exact."""
    def __init__(self,cfg):
        super().__init__(); self.config=cfg
        self.model=nn.Module(); text=self.model.language_model=nn.Module()
        text.embed_tokens=nn.Embedding(128,512,dtype=torch.float16)
        text.layers=nn.ModuleList()
        for _ in range(cfg.text_config.num_hidden_layers):
            layer=nn.Module(); layer.self_attn=nn.Module(); layer.mlp=nn.Module()
            layer.norm=nn.LayerNorm(128,dtype=torch.float16)
            for role,_ in runtime._ROLES:
                owner,name=role.split('.')
                getattr(layer,owner).add_module(name,nn.Linear(512,128,bias=False,dtype=torch.float16))
            text.layers.append(layer)
        self.model.visual=nn.Linear(128,128,bias=False,dtype=torch.float16)
        self.lm_head=nn.Linear(512,128,bias=False,dtype=torch.float16)


def tiny_schema(cfg):
    return ({f'{convert.TEXT_PREFIX}{i}.{role}.weight':(128,512)
             for i in range(cfg['text_config']['num_hidden_layers']) for role,_ in runtime._ROLES},(128,512))


def write_tiny_source(directory,cfg,*,explicit_head=False):
    directory.mkdir()
    with torch.device('meta'): model=TinyStorage(cfg)
    tensors={}
    for index,(name,value) in enumerate(model.named_parameters()):
        if name==convert.HEAD_NAME and not explicit_head: continue
        tensors[name]=((torch.arange(value.numel()).reshape(value.shape)%127)-63).half()/64 + (index%7)/32
    save_file(tensors,str(directory/'model.safetensors'))
    (directory/'config.json').write_text(cfg.to_json_string(use_diff=False))
    return tensors


def write_tiny_awq_source(directory, size=2):
    cfg = exact_config(size)
    tensors = write_tiny_source(directory, cfg, explicit_head=size == 4)
    tensors = {name: value.bfloat16() for name, value in tensors.items()}
    projections, _ = tiny_schema(cfg.to_dict())
    for index, (name, (n, k)) in enumerate(projections.items()):
        del tensors[name]
        q = ((torch.arange(n*k).reshape(n, k) + index) % 16 - 8).to(torch.int8)
        base = name[:-7]
        tensors[base + '.weight_packed'] = source_pack(q)
        tensors[base + '.weight_shape'] = torch.tensor([n, k])
        tensors[base + '.weight_scale'] = (
            (torch.arange(n*k//32).reshape(n, k//32) % 17 + 1).float() / 128).bfloat16()
    save_file(tensors, str(directory/'model.safetensors'))
    return cfg, tensors


@pytest.mark.parametrize('size', [2, 4])
@pytest.mark.parametrize('sharded', [False, True])
def test_awq_converter_preserves_calibrated_text_floats_and_loader_bytes(tmp_path, monkeypatch, size, sharded):
    from rpu_backend.quant.load_qwen3_vl_awq import _pack_projection

    src = tmp_path/'source'
    source_config, values = write_tiny_awq_source(src, size)
    if sharded:
        (src/'model.safetensors').unlink()
        # Source triples may span shards. Scales must follow their weight into
        # the output shard even when the original scale-only shard disappears.
        source_shards = {'model-00001-of-00002.safetensors': {}, 'model-00002-of-00002.safetensors': {}}
        for name, value in values.items():
            shard = list(source_shards)[int(name.endswith(('.weight_scale', '.weight_shape')))]
            source_shards[shard][name] = value
        index = {'metadata': {}, 'weight_map': {name: shard for shard, items in source_shards.items() for name in items}}
        for shard, items in source_shards.items(): save_file(items, str(src/shard))
        (src/'model.safetensors.index.json').write_text(json.dumps(index))
    monkeypatch.setattr(convert, '_checkpoint_tensor_shapes', tiny_schema)
    monkeypatch.setattr(runtime.AutoModelForImageTextToText, 'from_config', TinyStorage)
    original_quantize = convert.quantize_weight
    head_source = values[convert.EMBED_NAME if size == 2 else convert.HEAD_NAME].half()
    calls = []
    def quantize_only_head(value, *, bits):
        assert bits == 4 and torch.equal(value, head_source)
        calls.append('head')
        return original_quantize(value, bits=bits)
    monkeypatch.setattr(convert, 'quantize_weight', quantize_only_head)
    dst = tmp_path/'converted'
    stats = convert.convert_checkpoint(src, dst, bits=4)
    assert calls == ['head']
    assert stats['n_quantized'] == 1
    assert stats['n_preserved_text_projections'] == source_config.text_config.num_hidden_layers * 7
    cfg = Qwen3VLConfig.from_dict(json.loads((dst/'config.json').read_text()))
    assert cfg.quant_config == awq_runtime_config(size).quant_config
    assert not cfg.tie_word_embeddings and not cfg.text_config.tie_word_embeddings
    assert getattr(cfg, 'quantization_config', None) is None
    raw = {}
    for shard in dst.glob('*.safetensors'): raw.update(load_file(str(shard)))
    if sharded:
        index = json.loads((dst/'model.safetensors.index.json').read_text())
        assert set(index['weight_map']) == set(raw)
        assert index['metadata']['total_size'] == sum(t.numel()*t.element_size() for t in raw.values())
        assert all((dst/shard).is_file() for shard in index['weight_map'].values())
    with torch.inference_mode():
        model = runtime.load_qwen3_vl_runtime_imagetext(cfg, str(dst), local_files_only=True)
    assert runtime._validate_qwen3_vl_runtime_model(model)
    assert model.lm_head.weight is not model.model.language_model.embed_tokens.weight
    expected_head, expected_head_scale = original_quantize(head_source, bits=4)
    assert torch.equal(raw[convert.HEAD_NAME], expected_head)
    assert torch.equal(raw['lm_head.weight_scale'], expected_head_scale)
    for name, module, partition in runtime._projection_inventory(model):
        packed, shape, scale = (values[name + '.' + suffix] for suffix in ('weight_packed', 'weight_shape', 'weight_scale'))
        expected_weight, expected_scale = _pack_projection(packed, shape, scale, n=128, k=512, partition=partition)
        assert torch.equal(module.weight, expected_weight)
        assert torch.equal(module.weight_scale, expected_scale)
        assert torch.equal(raw[name + '.weight_scale'], scale.half().t())
    for name, value in values.items():
        if name.endswith(('.weight_packed', '.weight_shape', '.weight_scale')) or name == convert.HEAD_NAME:
            continue
        assert torch.equal(raw[name], value.half()), name
        assert torch.equal(model.get_parameter(name), value.half()), name
    assert not any(name.endswith(('.weight_packed', '.weight_shape')) for name in raw)
    # AWQ companion floats are also covered by the normal CPU ownership seal.
    model.model.language_model.layers[0].norm.weight.add_(1)
    with pytest.raises(RuntimeError, match='inventory'):
        runtime._validate_qwen3_vl_runtime_model(model)


@pytest.mark.parametrize('fault', ['bits8', 'config', 'missing_float', 'packed_dtype', 'scale_dtype',
                                  'float_dtype', 'shape_header', 'extra_tensor', 'wrong_index'])
def test_awq_conversion_rejects_wrong_config_headers_and_index_before_payload(tmp_path, monkeypatch, fault):
    src = tmp_path/'source'
    cfg, values = write_tiny_awq_source(src)
    if fault == 'config':
        cfg.quantization_config['config_groups']['group_0']['weights']['symmetric'] = False
        (src/'config.json').write_text(cfg.to_json_string(use_diff=False))
    elif fault == 'missing_float': del values['model.language_model.layers.0.norm.weight']
    elif fault == 'packed_dtype':
        name = f'{convert.TEXT_PREFIX}0.self_attn.q_proj.weight_packed'; values[name] = values[name].long()
    elif fault == 'scale_dtype':
        name = f'{convert.TEXT_PREFIX}0.self_attn.q_proj.weight_scale'; values[name] = values[name].float()
    elif fault == 'float_dtype': values[convert.EMBED_NAME] = values[convert.EMBED_NAME].float()
    elif fault == 'shape_header':
        name = f'{convert.TEXT_PREFIX}0.self_attn.q_proj.weight_shape'; values[name] = torch.tensor([128])
    elif fault == 'extra_tensor': values['foreign.weight'] = torch.ones(1).half()
    save_file(values, str(src/'model.safetensors'))
    if fault == 'wrong_index':
        # Header inventory is complete, but the index invents an extra tensor.
        mapping = {name: 'model.safetensors' for name in values}
        mapping['foreign.weight'] = 'model.safetensors'
        (src/'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
    monkeypatch.setattr(convert, '_checkpoint_tensor_shapes', tiny_schema)
    monkeypatch.setattr(runtime.AutoModelForImageTextToText, 'from_config', TinyStorage)
    monkeypatch.setattr(convert, '_awq_logical_projection', lambda *a, **k: pytest.fail('converted before header validation'))
    monkeypatch.setattr(convert, 'quantize_weight', lambda *a, **k: pytest.fail('quantized before header validation'))
    dst = tmp_path/'converted'
    with pytest.raises(ValueError): convert.convert_checkpoint(src, dst, bits=8 if fault == 'bits8' else 4)
    assert not dst.exists() and not list(tmp_path.glob('.converted.tmp-*'))


@pytest.mark.parametrize('fault', ['max_position_float', 'vision_position_float', 'missing_rope_type'])
def test_awq_conversion_rejects_geometry_unsupported_by_runtime_before_meta_or_payload(tmp_path, monkeypatch, fault):
    src = tmp_path/'source'
    write_tiny_awq_source(src)
    cfg = json.loads((src/'config.json').read_text())
    if fault == 'max_position_float':
        cfg['text_config']['max_position_embeddings'] = 262144.0
    elif fault == 'vision_position_float':
        cfg['vision_config']['num_position_embeddings'] = 2304.0
    else:
        cfg['text_config']['rope_parameters'].pop('rope_type')
    # The legacy AWQ source admits these values; derived runtime assets do not.
    assert load.is_qwen3_vl_awq_config(cfg)
    assert not convert.matches_runtime_geometry(cfg)
    (src/'config.json').write_text(json.dumps(cfg))
    monkeypatch.setattr(convert, '_checkpoint_layout', lambda *a, **k: pytest.fail('read before geometry validation'))
    monkeypatch.setattr(convert, '_awq_checkpoint_schema', lambda *a, **k: pytest.fail('meta model before geometry validation'))
    monkeypatch.setattr(convert, 'quantize_weight', lambda *a, **k: pytest.fail('quantized before geometry validation'))
    dst = tmp_path/'converted'
    with pytest.raises(ValueError, match='runtime geometry'):
        convert.convert_checkpoint(src, dst, bits=4)
    assert not dst.exists() and not list(tmp_path.glob('.converted.tmp-*'))


@pytest.mark.parametrize('fault', ['shape_value', 'nan_scale', 'zero_scale', 'underflow_scale', 'overflow_scale', 'nan_float'])
def test_awq_invalid_payload_cleans_only_its_incomplete_output(tmp_path, monkeypatch, fault):
    src = tmp_path/'source'
    _, values = write_tiny_awq_source(src)
    base = f'{convert.TEXT_PREFIX}0.self_attn.q_proj'
    if fault == 'shape_value': values[base + '.weight_shape'][0] = 127
    elif fault == 'nan_float': values[convert.EMBED_NAME].flatten()[0] = float('nan')
    else:
        values[base + '.weight_scale'].flatten()[0] = {
            'nan_scale': float('nan'), 'zero_scale': 0, 'underflow_scale': 1e-40, 'overflow_scale': 1e10}[fault]
    save_file(values, str(src/'model.safetensors'))
    monkeypatch.setattr(convert, '_checkpoint_tensor_shapes', tiny_schema)
    monkeypatch.setattr(runtime.AutoModelForImageTextToText, 'from_config', TinyStorage)
    dst = tmp_path/'converted'
    with pytest.raises(ValueError): convert.convert_checkpoint(src, dst, bits=4)
    assert not dst.exists() and not list(tmp_path.glob('.converted.tmp-*'))
    assert (src/'model.safetensors').is_file()


@pytest.mark.parametrize('size,bits',[(2,8),(2,4),(4,4)])
def test_real_converter_to_cpu_loader_math_inventory_and_independent_head(tmp_path,monkeypatch,size,bits):
    source=write_tiny_source(tmp_path/'source',config(size,bits,source=True),explicit_head=size==4)
    monkeypatch.setattr(convert,'_checkpoint_tensor_shapes',tiny_schema)
    monkeypatch.setattr(runtime.AutoModelForImageTextToText,'from_config',TinyStorage)
    destination=tmp_path/'converted'
    stats=convert.convert_checkpoint(tmp_path/'source',destination,bits=bits)
    cfg=Qwen3VLConfig.from_dict(json.loads((destination/'config.json').read_text()))
    assert stats['n_quantized']==cfg.text_config.num_hidden_layers*7+1
    with torch.inference_mode():
        model=runtime.load_qwen3_vl_runtime_imagetext(cfg,str(destination),local_files_only=True)
    assert runtime._validate_qwen3_vl_runtime_model(model)
    assert not model.config.tie_word_embeddings and not model.config.text_config.tie_word_embeddings
    assert model.lm_head.weight is not model.model.language_model.embed_tokens.weight
    assert model.lm_head.weight.data_ptr()!=model.model.language_model.embed_tokens.weight.data_ptr()
    assert model.lm_head.weight._version>=0
    raw=load_file(str(destination/'model.safetensors'))
    for name,module,partition in runtime._quantized_modules(model):
        src=source.get(name+'.weight',source[convert.EMBED_NAME])
        expected_weight,expected_scale=convert.quantize_weight(src,bits=bits)
        assert torch.equal(raw[name+'.weight'],expected_weight)
        assert torch.equal(raw[name+'.weight_scale'],expected_scale)
        if bits==8:
            assert torch.equal(module.weight,expected_weight) and torch.equal(module.weight_scale,expected_scale)
            assert not hasattr(module,'_rpu_linear_partition')
        else:
            q,s,_=quantize_int4_group_wise(src,32)
            assert torch.equal(inverse_rpu_weight(module.weight,128,512,partition),q)
            assert torch.equal(inverse_rpu_scale(module.weight_scale,128,512,partition),s.t())
            assert module._rpu_linear_partition==partition and module._rpu_linear_num_cores==8
    assert torch.equal(model.model.language_model.embed_tokens.weight,source[convert.EMBED_NAME])
    assert torch.equal(model.model.visual.weight,source['model.visual.weight'])


@pytest.mark.parametrize('fault',[None,'missing_embed','wrong_shard','extra_index',
                                  'unindexed_tensor','duplicate_tensor','projection_shape'])
def test_converter_checks_actual_shard_headers_before_reading_weights(tmp_path,monkeypatch,fault):
    source=tmp_path/'source'
    values=write_tiny_source(source,config(2,4,source=True),explicit_head=True)
    (source/'model.safetensors').unlink()
    first='model-00001-of-00002.safetensors'
    second='model-00002-of-00002.safetensors'
    shards={first:{convert.HEAD_NAME:values[convert.HEAD_NAME]},
            second:{name:value for name,value in values.items() if name!=convert.HEAD_NAME}}
    declared={name:shard for shard,tensors in shards.items() for name in tensors}
    if fault=='missing_embed': del shards[second][convert.EMBED_NAME]
    elif fault=='wrong_shard':
        declared[convert.HEAD_NAME]=second
        declared[convert.EMBED_NAME]=first
    elif fault=='extra_index': declared['foreign.weight']=second
    elif fault=='unindexed_tensor': shards[second]['foreign.weight']=torch.ones(1).half()
    elif fault=='duplicate_tensor': shards[second][convert.HEAD_NAME]=values[convert.HEAD_NAME]
    elif fault=='projection_shape':
        name=f'{convert.TEXT_PREFIX}0.self_attn.q_proj.weight'
        shards[second][name]=shards[second][name][:-1].contiguous()
    for name,tensors in shards.items(): save_file(tensors,str(source/name))
    (source/'model.safetensors.index.json').write_text(json.dumps({'metadata':{},'weight_map':declared}))
    monkeypatch.setattr(convert,'_checkpoint_tensor_shapes',tiny_schema)
    destination=tmp_path/'converted'
    if fault is not None:
        monkeypatch.setattr(convert,'load_file',lambda *a,**k:pytest.fail('read source payload before header validation'))
        with pytest.raises((ValueError,KeyError),match='source|header'):
            convert.convert_checkpoint(source,destination,bits=4)
        assert not destination.exists() and not list(tmp_path.glob('.converted.tmp-*'))
        return
    result=convert.convert_checkpoint(source,destination,bits=4)
    assert result['n_quantized']==197
    output_index=json.loads((destination/'model.safetensors.index.json').read_text())['weight_map']
    expected_text,_=tiny_schema(json.loads((source/'config.json').read_text()))
    for shard,tensors in shards.items():
        converted=load_file(str(destination/shard))
        for name,value in tensors.items():
            assert output_index[name]==shard
            if name in expected_text or name==convert.HEAD_NAME:
                weight,scale=convert.quantize_weight(value,bits=4)
                assert torch.equal(converted[name],weight)
                assert torch.equal(converted[name[:-7]+'.weight_scale'],scale)
            else:
                assert torch.equal(converted[name],value.half())


@pytest.mark.parametrize('fault',['missing','extra','weight_shape','weight_dtype','scale_dtype','float_dtype'])
def test_headers_rejected_before_unpacking_or_parameter_install(tmp_path,monkeypatch,fault):
    cfg=config(); (tmp_path/'config.json').write_text(cfg.to_json_string(use_diff=False))
    values={}
    if fault=='extra': values['foreign.weight']=torch.ones(1).half()
    elif fault=='weight_shape': values['lm_head.weight']=torch.zeros(1,1,dtype=torch.uint8)
    elif fault=='weight_dtype': values['lm_head.weight']=torch.zeros(128,256,dtype=torch.int8)
    elif fault=='scale_dtype': values['lm_head.weight_scale']=torch.ones(16,128,dtype=torch.float32)
    elif fault=='float_dtype': values[convert.EMBED_NAME]=torch.zeros(128,512,dtype=torch.bfloat16)
    save_file(values,str(tmp_path/'model.safetensors'))
    monkeypatch.setattr(runtime.AutoModelForImageTextToText,'from_config',TinyStorage)
    monkeypatch.setattr(runtime,'_pack_w4_projection',lambda *a,**k:pytest.fail('unpacked incomplete inventory'))
    with pytest.raises(ValueError,match='metadata|inventory|Unexpected'):
        runtime.load_qwen3_vl_runtime_imagetext(cfg,str(tmp_path),local_files_only=True)


def sealed_storage(bits=4,size=2):
    # Reduced layer count only for mutation/migration helper tests, not loader admission.
    cfg=config(size,bits)
    model=TinyStorage(cfg)
    model.model.language_model.layers=nn.ModuleList(list(model.model.language_model.layers)[:1])
    for _,module,partition in runtime._quantized_modules(model):
        weight,scale=convert.quantize_weight(torch.ones(128,512),bits=bits)
        if bits==4:
            weight,scale=runtime._pack_w4_projection(weight,scale,n=128,k=512,partition=partition)
            module._rpu_linear_partition=partition;module._rpu_linear_num_cores=8
        module.weight=nn.Parameter(weight,requires_grad=False)
        module.register_buffer('weight_scale',scale,persistent=False)
    model.requires_grad_(False)
    setattr(model,runtime._PLAN_ATTR,runtime._RuntimeLoadPlan(id(model),runtime._config_signature(cfg),
        runtime._inventory(model),runtime._module_inventory(model)))
    return model


@pytest.mark.parametrize('fault',['owner','config','head_weight','head_scale','head_layout','last_weight','missing_seal'])
def test_cpu_seal_rejects_drift_before_any_device_allocation(monkeypatch,fault):
    model=sealed_storage(); assert runtime._validate_qwen3_vl_runtime_model(model)
    if fault=='owner': setattr(model,runtime._PLAN_ATTR,replace(getattr(model,runtime._PLAN_ATTR),model_id=-1))
    elif fault=='config': model.config.text_config.rms_norm_eps=1e-5
    elif fault=='head_weight': model.lm_head.weight.add_(1)
    elif fault=='head_scale': model.lm_head.weight_scale=model.lm_head.weight_scale.clone()
    elif fault=='head_layout': model.lm_head._rpu_linear_partition=0
    elif fault=='last_weight': model.model.language_model.layers[-1].mlp.down_proj.weight.add_(1)
    else: delattr(model,runtime._PLAN_ATTR)
    model._rpu_swizzle_started=True
    monkeypatch.setattr(load,'_allocate_decoder_projection_views',lambda *a,**k:pytest.fail('early allocation'))
    with pytest.raises(RuntimeError): runtime._move_materialized_runtime_decoder_for_rpu(model)


@pytest.mark.parametrize('size',[2,4])
def test_w4_migration_reuses_weight_scale_banks_without_reswizzle_and_keeps_head(monkeypatch,size):
    model=sealed_storage(size=size)
    with pytest.raises(RuntimeError,match='ownership'): runtime._move_materialized_runtime_decoder_for_rpu(model)
    model._rpu_swizzle_started=True
    head=(model.lm_head.weight,model.lm_head.weight_scale)
    source=[(m.weight.clone(),m.weight_scale.clone()) for _,m,_ in runtime._projection_inventory(model)]
    original=torch.empty;allocations=[]
    def empty(*a,**k):
        if k.get('device')=='rpu': allocations.append(k['dtype']);k['device']='cpu'
        return original(*a,**k)
    monkeypatch.setattr(torch,'empty',empty)
    monkeypatch.setattr(nn.Module,'to',lambda self,*a,**k:self)
    from rpu_backend.runtime import weights
    monkeypatch.setattr(weights,'convert_linear_weights_inplace',lambda *a,**k:pytest.fail('second W4 swizzle'))
    runtime._move_materialized_runtime_decoder_for_rpu(model)
    assert allocations==[torch.uint8,torch.float16]
    for (_,module,_),(weight,scale) in zip(runtime._projection_inventory(model),source):
        assert torch.equal(module.weight,weight) and torch.equal(module.weight_scale,scale)
        assert module.weight_scale.storage_offset()*2%4096==0
    assert model.lm_head.weight is head[0] and model.lm_head.weight_scale is head[1]
    with pytest.raises(RuntimeError,match='inventory'): runtime._move_materialized_runtime_decoder_for_rpu(model)


def test_new_w8_migration_reuses_existing_single_swizzle_bank_path(monkeypatch):
    model=sealed_storage(bits=8);model._rpu_swizzle_started=True;called=[]
    monkeypatch.setattr(load,'_move_materialized_decoder_for_rpu',lambda m,**k:called.append((m,k)))
    runtime._move_materialized_runtime_decoder_for_rpu(model)
    assert called==[(model,{'dtype':torch.int8})]


@pytest.mark.parametrize('size',[2,4])
def test_awq_requantization_to_w8_is_rejected_before_reading_weights(tmp_path,size):
    (tmp_path/'config.json').write_text(exact_config(size).to_json_string(use_diff=False))
    with pytest.raises(ValueError,match='bits=4'):
        convert.convert_checkpoint(tmp_path,tmp_path/'destination',bits=8)
    assert not (tmp_path/'destination').exists()
