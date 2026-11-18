"""Quantized geometry admission must not widen FP16 or accept foreign payloads."""
from types import SimpleNamespace
import copy
import pytest
import torch
from transformers import Qwen3Config
from rpu_backend.quant.qwen3_profiles import qwen3_dense_quant_profile, is_qwen3_32b_w8a16_config, qwen3_w4_topology
from rpu_backend.runtime.topology import (decoder_geometry_profile, resolve_decoder_topology,
    decoder_mlp_intermediate_size, decoder_packed_w4_mlp_intermediate_size)
from rpu_backend.runtime.decoder import _validate_decoder_w4_scale_lists

GEOMETRIES = [(1024,3072,28,16), (2048,6144,28,16), (2560,9728,36,32),
              (4096,12288,36,32), (5120,17408,40,40), (5120,25600,64,64)]

def config(geometry):
    h,i,l,q=geometry
    return Qwen3Config(architectures=['Qwen3ForCausalLM'], hidden_size=h,
        intermediate_size=i,num_hidden_layers=l,num_attention_heads=q,
        num_key_value_heads=8,head_dim=128,vocab_size=151936,
        tie_word_embeddings=h<4096,rms_norm_eps=1e-6,max_position_embeddings=40960)

@pytest.mark.parametrize('geometry',GEOMETRIES)
def test_exact_source_geometry_resolves_only_private_quant_profile(geometry):
    c=config(geometry)
    profile=qwen3_dense_quant_profile(c)
    assert profile.hidden_size==geometry[0] and profile.num_layers==geometry[2]
    assert resolve_decoder_topology(num_cores=8,**profile.geometry()).num_cores==8
    assert decoder_geometry_profile(**profile.geometry()) is not None
    c.num_hidden_layers += 1
    with pytest.raises(ValueError): qwen3_dense_quant_profile(c)


@pytest.mark.parametrize('geometry', GEOMETRIES)
@pytest.mark.parametrize('cores', [4, 6, 8])
def test_w4_topology_respects_budget_and_packed_row_alignment(geometry, cores):
    topology = qwen3_w4_topology(config(geometry), cores)
    h, intermediate, _, q = geometry
    assert topology.num_cores == cores
    assert topology.attn_tp == (8 if cores == 8 else 4)
    assert topology.lm_head_tp == cores
    # Packed W4 shares the FP16/W8 template: MLP6 at budget 6, never MLP4.
    assert topology.mlp_tp == cores
    assert topology == resolve_decoder_topology(
        num_cores=cores, **qwen3_dense_quant_profile(config(geometry)).geometry())
    physical = decoder_packed_w4_mlp_intermediate_size(intermediate, topology.mlp_tp)
    assert physical % (32 * topology.mlp_tp) == 0 and 0 <= physical - intermediate < 192
    assert q * 128 % (64 * topology.attn_tp) == 0
    expected = {9728: 9792, 17408: 17472, 25600: 25728}.get(intermediate, intermediate)
    assert physical == (expected if cores == 6 else intermediate)
    # FP16/W8 six-lane padding is unchanged by the packed W4 width.
    assert decoder_mlp_intermediate_size(intermediate, 6) == {
        9728: 9792, 17408: 17472, 25600: 25728}.get(intermediate, intermediate)

@pytest.mark.parametrize('change',[
    {'quant_config':{'method':'w4a16'}}, {'quantization_config':{'quant_method':'gptq'}},
    {'attention_bias':True}, {'num_attention_heads':True}, {'hidden_act':'gelu'},
    {'max_position_embeddings':None}, {'tie_word_embeddings':True},
    {'layer_types':['sliding_attention']*64}, {'rope_parameters':{'rope_type':'yarn'}},
])
def test_w4_rejects_foreign_precision_and_semantics(change):
    c=config(GEOMETRIES[-1])
    # Exercise backend admission even when newer HF configs reject assignment.
    for k,v in change.items(): object.__setattr__(c,k,v)
    with pytest.raises(ValueError): qwen3_dense_quant_profile(c)

def test_w8_32b_requires_actual_converter_recipe_and_never_is_a_w4_source():
    c=config(GEOMETRIES[-1])
    c.quant_config={'method':'w8a16','mode':'per_channel_symmetric','qaxis':0,
        'skip_modules':[],'quantized_lm_head':True,'quantized_embed_tokens':False,
        'lm_head_untied':True,'embed_tokens_untied':False}
    assert is_qwen3_32b_w8a16_config(c)
    with pytest.raises(ValueError): qwen3_dense_quant_profile(c)
    for key,value in [('qaxis',False),('quantized_lm_head',False),('extra',True)]:
        wrong=copy.deepcopy(c)
        wrong.quant_config[key]=value
        assert not is_qwen3_32b_w8a16_config(wrong)

@pytest.mark.parametrize('geometry',GEOMETRIES)
@pytest.mark.parametrize('cores', [4, 6, 8])
def test_packed_payload_validates_shape_scale_identity_and_geometry(geometry, cores):
    c=config(geometry)
    p=qwen3_dense_quant_profile(c)
    topology = qwen3_w4_topology(c, cores)
    h=p.hidden_size
    i=decoder_packed_w4_mlp_intermediate_size(p.intermediate_size, topology.mlp_tp)
    q,k=p.num_q_heads*128,1024
    roles=[('self_attn.q_proj',q,h,1),('self_attn.k_proj',k,h,1),
           ('self_attn.v_proj',k,h,1),('self_attn.o_proj',h,q,0),
           ('mlp.gate_proj',i,h,1),('mlp.up_proj',i,h,1),('mlp.down_proj',h,i,0)]
    layer=torch.nn.Module()
    layer.self_attn=torch.nn.Module()
    layer.mlp=torch.nn.Module()
    scales=[]
    for role,n,k,col in roles:
        module=torch.nn.Linear(k,n,bias=False,device='meta')
        module.weight=torch.nn.Parameter(torch.empty(n,k//2,device='meta',dtype=torch.uint8),requires_grad=False)
        tp = topology.mlp_tp if role.startswith('mlp.') else topology.attn_tp
        lg,ln=(k//32,n//tp) if col else (k//32//tp,n)
        count=tp*((lg+3)//4*4)*((ln+63)//64*64)
        module.register_buffer('weight_scale',torch.empty(32,count//32,device='meta',dtype=torch.float16))
        parent,name=role.split('.')
        setattr(getattr(layer,parent),name,module)
        scales.append([module.weight_scale]*p.num_layers)
    model=SimpleNamespace(config=c,layers=[layer]*p.num_layers)
    _validate_decoder_w4_scale_lists(model,scales,topology)
    scales[0]=[s.clone() for s in scales[0]]
    with pytest.raises(ValueError,match='actual UINT8/G32'):
        _validate_decoder_w4_scale_lists(model,scales,topology)

def test_installed_b1_profile_rejects_cache_batch_before_allocation():
    from rpu_backend.api.cache import RPUCache
    owner=SimpleNamespace(config=config(GEOMETRIES[-1]),
                          model=SimpleNamespace(_rpu_batch_decode_enabled=False))
    with pytest.raises(ValueError,match='batch_size=1'):
        RPUCache.from_model(owner,max_seq_len=128,batch_size=2)


@pytest.mark.parametrize('geometry', GEOMETRIES)
@pytest.mark.parametrize('cores', [4, 6, 8])
def test_packed_head_handoff_uses_cold_head_partition(monkeypatch, geometry, cores):
    from rpu_backend.adapters import qwen3
    from rpu_backend.runtime.topology import decoder_lm_head_vocab_size
    c = config(geometry)
    topology = qwen3_w4_topology(c, cores)
    tp = topology.lm_head_tp
    padded_vocab = decoder_lm_head_vocab_size(c.vocab_size, tp)
    head = torch.nn.Linear(c.hidden_size, padded_vocab, bias=False, device='meta')
    from rpu_backend.quant.int4_pgrp_pack import w4_storage_shape
    head.weight = torch.nn.Parameter(torch.empty(
        *w4_storage_shape(padded_vocab, c.hidden_size, 1, tp),
        dtype=torch.uint8, device='meta'), requires_grad=False)
    count = tp * ((c.hidden_size // 32 + 3) // 4) * ((padded_vocab // tp + 63) // 64) * 256
    head.register_buffer('weight_scale', torch.empty(32, count // 32,
        dtype=torch.float16, device='meta'))
    head._rpu_linear_partition, head._rpu_linear_num_cores = 1, tp
    model = SimpleNamespace(config=c, lm_head=head,
        to=SimpleNamespace(__rpu_quantization__='w4a16_lm_head'),
        _rpu_execution={'model': {'num_cores': cores}},
        model=SimpleNamespace(_rpu_decoder_handle=7, _rpu_decoder_topology=topology))
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, 'to', lambda value, *args, **kwargs:
        value if args and args[0] == 'rpu' else original_to(value, *args, **kwargs))
    monkeypatch.setattr(qwen3, 'transform_linear_weight', lambda *a, **k: pytest.fail('packed head reswizzle'))
    published = []
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_set_lm_head',
        lambda *args: published.append(args), raising=False)
    qwen3._push_lm_head_weight(model, 7)
    assert len(published) == 1 and published[0][0] == 7
    assert published[0][1].dtype == torch.uint8
    assert published[0][2].shape == (32, count // 32)
    head._rpu_linear_num_cores = 8 if tp == 4 else 4
    with pytest.raises(qwen3.RPUBackendError, match='G32'):
        qwen3._push_lm_head_weight(model, 7)
