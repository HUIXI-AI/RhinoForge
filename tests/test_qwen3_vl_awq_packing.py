"""AWQ byte semantics, invalid payloads and cold ownership; no hardware needed."""
import copy
import gc
from dataclasses import replace

import pytest
import torch
from torch import nn
from transformers import Qwen3VLConfig

from rpu_backend.quant import load
from rpu_backend.quant import load_qwen3_vl_awq as awq


def exact_config(size=2):
    h, inter, layers, heads = (2048, 6144, 28, 16) if size == 2 else (2560, 9728, 36, 32)
    ignored = [f"model.visual.blocks.{i}.{role}" for i in range(24)
               for role in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")]
    ignored += [f"model.visual.{owner}.linear_fc{fc}" for owner in
                ("merger", "deepstack_merger_list.0", "deepstack_merger_list.1", "deepstack_merger_list.2") for fc in (1, 2)]
    return Qwen3VLConfig(
        architectures=["Qwen3VLForConditionalGeneration"], tie_word_embeddings=size == 2,
        text_config=dict(hidden_size=h, intermediate_size=inter, num_hidden_layers=layers,
                         num_attention_heads=heads, num_key_value_heads=8, head_dim=128,
                         vocab_size=151936, hidden_act="silu", rms_norm_eps=1e-6,
                         attention_bias=False, attention_dropout=0., max_position_embeddings=262144,
                         use_cache=True, tie_word_embeddings=True,
                         rope_parameters=dict(rope_type="default", rope_theta=5000000,
                                              mrope_interleaved=True, mrope_section=[24,20,20])),
        vision_config=dict(hidden_size=1024, intermediate_size=4096, depth=24,
                           num_heads=16, patch_size=16, temporal_patch_size=2,
                           spatial_merge_size=2, out_hidden_size=h, hidden_act="gelu_pytorch_tanh",
                           in_channels=3, num_position_embeddings=2304, deepstack_visual_indexes=[5,11,17]),
        image_token_id=151655, video_token_id=151656, vision_start_token_id=151652, vision_end_token_id=151653,
        quantization_config=dict(quant_method="compressed-tensors", format="pack-quantized",
            quantization_status="compressed", kv_cache_scheme=None, sparsity_config={}, transform_config={},
            ignore=ignored+["lm_head"], config_groups={"group_0": dict(format="pack-quantized", targets=["Linear"],
                input_activations=None, output_activations=None, weights=dict(num_bits=4, group_size=32,
                    symmetric=True, dynamic=False, strategy="group", type="int", actorder=None, block_structure=None))}))


def source_pack(q):
    # Official CT format is offset by8, low K index in the low nibble.
    q = (q.to(torch.int64)+8).view(q.shape[0], -1, 8)
    return torch.sum(q << (torch.arange(8)*4), dim=-1).to(torch.int32)


def inverse_rpu_weight(weight, n, k, partition):
    b = weight.flatten().to(torch.int16)
    q = torch.stack((b & 15, b >> 4), dim=-1).flatten()
    q = torch.where(q >= 8, q-16, q).to(torch.int8)
    order = torch.tensor([v for base in (0, 32) for i in range(16) for v in (base+i, base+i+16)])
    if partition == 1:
        q = q.view(n//128, k//64, 8, 16, 64)[..., order.argsort()]
        return q.permute(2,0,3,1,4).reshape(n,k)
    q = q.view(n//16, k//512, 8, 16, 64)[..., order.argsort()]
    return q.permute(0,3,2,1,4).reshape(n,k)


def inverse_rpu_scale(scale, n, k, partition):
    g = k//32
    lg, ln = (g,n//8) if partition == 1 else (g//8,n)
    gb, nb = (lg+3)//4,(ln+63)//64
    local = scale.view(gb,nb,8,4,64).permute(2,0,3,1,4).reshape(8,gb*4,nb*64)
    assert torch.count_nonzero(local[:,lg:,:]) == 0
    assert torch.count_nonzero(local[:,:lg,ln:]) == 0
    real = local[:,:lg,:ln]
    return (real.permute(1,0,2).reshape(g,n) if partition == 1 else real.reshape(g,n)).t().contiguous()


@pytest.mark.parametrize("partition,n,k", [(1,128,512), (0,128,9728)])
def test_actual_offset_nibbles_and_tp8_pgrp_scale_padding_roundtrip(partition,n,k):
    q = ((torch.arange(n*k).reshape(n,k)*5+torch.arange(n)[:,None]*3)%16-8).to(torch.int8)
    scales = (torch.arange(n*(k//32)).reshape(n,-1)%47+1).to(torch.bfloat16)/256
    source = source_pack(q)
    weight, scale = awq._pack_projection(source, torch.tensor([n,k]), scales,n=n,k=k,partition=partition)
    assert weight.dtype == torch.uint8 and tuple(weight.shape)==(n,k//2)
    assert scale.dtype == torch.float16 and scale.size(0)==32
    assert torch.equal(inverse_rpu_weight(weight,n,k,partition), q)
    assert torch.equal(inverse_rpu_scale(scale,n,k,partition),scales.half())
    assert torch.equal(source,source_pack(q))


@pytest.mark.parametrize("fault", ["zero", "nan", "overflow", "shape_values", "packed_dtype", "scale_shape"])
def test_bad_payload_fails_before_physical_packing(monkeypatch,fault):
    n,k=128,512
    packed=torch.zeros(n,k//8,dtype=torch.int32)
    shape=torch.tensor([n,k]);scale=torch.ones(n,k//32,dtype=torch.bfloat16)
    if fault=="zero":scale[0,0]=0
    elif fault=="nan":scale[0,0]=float('nan')
    elif fault=="overflow":scale[0,0]=1e6
    elif fault=="shape_values":shape[1]=1024
    elif fault=="packed_dtype":packed=packed.long()
    else:scale=scale[:,:-1]
    monkeypatch.setattr(awq,"swizzle_pack_int4_pgrp",lambda *a,**k:pytest.fail("packed invalid input"))
    with pytest.raises(ValueError):awq._pack_projection(packed,shape,scale,n=n,k=k,partition=1)


@pytest.mark.parametrize("size", [2,4])
def test_exact_profile_and_neighbor_rejection(size):
    c=exact_config(size);assert awq.is_qwen3_vl_awq_config(c)
    for key,value in [("group_size",64),("symmetric",False),("actorder","group"),("num_bits",8),("dynamic",True),("zero_point",1),("group_size",32.0)]:
        bad=copy.deepcopy(c);bad.quantization_config['config_groups']['group_0']['weights'][key]=value
        assert not awq.is_qwen3_vl_awq_config(bad),key
    for change in ('tie','ignore','activation','geometry','rope','vision'):
        bad=copy.deepcopy(c)
        if change=='tie':bad.tie_word_embeddings=not bad.tie_word_embeddings
        elif change=='ignore':bad.quantization_config['ignore'].remove('lm_head')
        elif change=='activation':bad.quantization_config['config_groups']['group_0']['input_activations']={'num_bits':8}
        elif change=='geometry':bad.text_config.num_key_value_heads=4
        elif change=='rope':bad.text_config.rope_parameters['mrope_interleaved']=False
        else:bad.vision_config.out_hidden_size+=128
        assert not awq.is_qwen3_vl_awq_config(bad),change


def sealed_small_model(size=2):
    """Small storage fixture for ownership checks, not an exact-model loader test."""
    m=nn.Module();m.config=exact_config(size);m.model=nn.Module();t=m.model.language_model=nn.Module()
    t.layers=nn.ModuleList()
    t.embed_tokens=nn.Embedding(16,128,dtype=torch.float16)
    m.lm_head=nn.Linear(128,16,bias=False,dtype=torch.float16);m.lm_head.weight=t.embed_tokens.weight
    for index in range(2):
        layer=nn.Module();layer.self_attn=nn.Module();layer.mlp=nn.Module()
        layer.norm=nn.LayerNorm(128,dtype=torch.float16)
        for role,part in awq._ROLES:
            parent,name=role.split('.')
            mod=nn.Linear(512,128,bias=False,dtype=torch.float16)
            mod.weight=nn.Parameter(torch.full((128,256),index+17,dtype=torch.uint8),requires_grad=False)
            mod.register_buffer('weight_scale',torch.arange(32*64).half().reshape(32,64),persistent=False)
            mod._rpu_linear_partition=part;mod._rpu_linear_num_cores=8
            getattr(layer,parent).add_module(name,mod)
        t.layers.append(layer)
    m.requires_grad_(False)
    plan=awq._AWQLoadPlan(id(m),awq._config_signature(m.config),awq._inventory(m),tuple(
        (name,id(mod),mod.in_features,mod.out_features,part,8) for name,mod,part in awq._projection_inventory(m)))
    setattr(m,awq._PLAN_ATTR,plan)
    return m


@pytest.mark.parametrize("fault", ['config','replacement','inplace','last_scale','layout','owner','ordinary'])
@pytest.mark.parametrize("size", [2, 4])
def test_changed_cpu_owner_inventory_is_rejected_before_allocation(monkeypatch,fault,size):
    m=sealed_small_model(size);assert awq._validate_qwen3_vl_awq_model(m)
    last=m.model.language_model.layers[-1].mlp.down_proj
    if fault=='config':m.config.text_config.rms_norm_eps=1e-5
    elif fault=='replacement':last.weight=nn.Parameter(last.weight.clone(),requires_grad=False)
    elif fault=='inplace':last.weight.add_(1)
    elif fault=='last_scale':last.weight_scale=last.weight_scale.clone()
    elif fault=='layout':last._rpu_linear_partition=1
    elif fault=='owner':setattr(m,awq._PLAN_ATTR,replace(getattr(m,awq._PLAN_ATTR),model_id=-1))
    else:delattr(m,awq._PLAN_ATTR)
    m._rpu_swizzle_started=True
    monkeypatch.setattr(load,'_allocate_decoder_projection_views',lambda *a,**k:pytest.fail('early allocation'))
    with pytest.raises(RuntimeError):awq._move_materialized_awq_decoder_for_rpu(m)


@pytest.mark.parametrize("size", [2, 4])
def test_packed_bank_copies_once_without_generic_swizzle_and_retains_storage(monkeypatch,size):
    m=sealed_small_model(size);source=[mod.weight.clone() for _,mod,_ in awq._projection_inventory(m)]
    scales=[mod.weight_scale.clone() for _,mod,_ in awq._projection_inventory(m)]
    with pytest.raises(RuntimeError,match='ownership'):awq._move_materialized_awq_decoder_for_rpu(m)
    m._rpu_swizzle_started=True
    original=torch.empty;alloc=[]
    def empty(*a,**kw):
        if kw.get('device')=='rpu':kw['device']='cpu';alloc.append((a,kw))
        return original(*a,**kw)
    monkeypatch.setattr(torch,'empty',empty)
    monkeypatch.setattr(nn.Module,'to',lambda self,*a,**kw:self)
    from rpu_backend.runtime import weights
    monkeypatch.setattr(weights,'convert_linear_weights_inplace',lambda *a,**kw:pytest.fail('double swizzle'))
    with torch.inference_mode():
        awq._move_materialized_awq_decoder_for_rpu(m)
        assert torch.is_inference_mode_enabled()
    assert len(alloc)==(2 if size==4 else 1)
    storage=set()
    scale_storage=set()
    for (_,mod,_),weight,scale in zip(awq._projection_inventory(m),source,scales):
        assert torch.equal(mod.weight,weight) and torch.equal(mod.weight_scale,scale)
        assert mod.weight._version >= 0
        assert mod.weight.storage_offset()%256==0
        storage.add(mod.weight.untyped_storage().data_ptr())
        scale_storage.add(mod.weight_scale.untyped_storage().data_ptr())
        assert 'weight_scale' in mod._buffers and 'weight_scale' in mod._non_persistent_buffers_set
        if size==4:
            assert mod.weight_scale.storage_offset()*2%4096==0
    assert len(storage)==1
    assert len(scale_storage)==(1 if size==4 else 14)
    assert m.lm_head.weight is m.model.language_model.embed_tokens.weight
    retained=m.model.language_model.layers[-1].mlp.down_proj.weight
    retained_scale=m.model.language_model.layers[-1].mlp.down_proj.weight_scale
    del m,mod;gc.collect();assert torch.equal(retained,source[-1])
    assert torch.equal(retained_scale,scales[-1])


def test_scale_bank_preserves_striped_padding_and_offsets_without_host_alignment_guess(monkeypatch):
    m=sealed_small_model(4)
    modules=list(awq._projection_inventory(m))
    # Exercise a non-4096-multiple payload and zero-filled physical group tails.
    for index,(_,mod,_) in enumerate(modules):
        mod.weight_scale=torch.arange(32*(80+index*16)).half().reshape(32,-1)
        mod.weight_scale[:,-16:]=0
    originals={name:mod.weight_scale.clone() for name,mod,_ in modules}
    original=torch.empty;alloc=[]
    def empty(*args,**kwargs):
        assert kwargs=={'dtype':torch.float16,'device':'rpu'}
        # Deliberately offset host storage. Production must never try to make
        # this host pointer stand in for RpuGetDevAddr's native alignment check.
        value=original(args[0]+1,dtype=torch.float16)[1:]
        alloc.append(value)
        return value
    monkeypatch.setattr(torch,'empty',empty)
    views=awq._allocate_awq_scale_views(m)
    assert len(alloc)==1
    base=alloc[0].data_ptr();end=0
    for name,mod,_ in modules:
        expected=(end+4095)//4096*4096
        assert views[name].data_ptr()-base==expected
        assert torch.equal(views[name],originals[name])
        assert torch.equal(mod.weight_scale,originals[name])
        end=expected+views[name].numel()*2
    assert alloc[0].numel()*2==end


@pytest.mark.parametrize("fault", ['dtype','shape','noncontiguous'])
def test_scale_bank_validates_last_scale_before_first_allocation(monkeypatch,fault):
    m=sealed_small_model(4);mod=m.model.language_model.layers[-1].mlp.down_proj
    if fault=='dtype':mod.weight_scale=mod.weight_scale.float()
    elif fault=='shape':mod.weight_scale=mod.weight_scale.flatten()
    else:mod.weight_scale=mod.weight_scale.t()
    monkeypatch.setattr(torch,'empty',lambda *a,**kw:pytest.fail('early scale allocation'))
    with pytest.raises(ValueError,match='contiguous CPU FP16'):
        awq._allocate_awq_scale_views(m)


@pytest.mark.parametrize("fault", ['allocation','last_copy'])
def test_4b_scale_bank_failure_keeps_original_buffers_and_install_poison(monkeypatch,fault):
    m=sealed_small_model(4);m._rpu_swizzle_started=True
    modules=list(awq._projection_inventory(m))
    owners=[(mod.weight,mod.weight_scale) for _,mod,_ in modules]
    original_empty=torch.empty;original_copy=torch.Tensor.copy_;copies=0
    def empty(*args,**kwargs):
        if kwargs.get('device')=='rpu':
            if kwargs['dtype']==torch.float16 and fault=='allocation':
                raise RuntimeError('scale bank allocation failed')
            kwargs['device']='cpu'
        return original_empty(*args,**kwargs)
    def copy_(target,source,*args,**kwargs):
        nonlocal copies
        if source.dtype==torch.float16 and tuple(source.shape)==(32,64):
            copies+=1
            if fault=='last_copy' and copies==len(modules):
                raise RuntimeError('scale bank copy failed')
        return original_copy(target,source,*args,**kwargs)
    monkeypatch.setattr(torch,'empty',empty)
    monkeypatch.setattr(torch.Tensor,'copy_',copy_)
    monkeypatch.setattr(nn.Module,'to',lambda *a,**kw:pytest.fail('moved layer before scale bank completed'))
    with pytest.raises(RuntimeError,match='scale bank .* failed'):
        awq._move_materialized_awq_decoder_for_rpu(m)
    assert m._rpu_swizzle_started
    assert all(mod.weight is weight and mod.weight_scale is scale
               for (_,mod,_),(weight,scale) in zip(modules,owners))


@pytest.mark.parametrize('bad_value',[float('nan'),float('inf'),1e10])
def test_float_constants_reject_nonfinite_or_fp16_overflow(bad_value):
    with pytest.raises(ValueError):awq._checked_float('norm',torch.tensor([bad_value],dtype=torch.bfloat16))


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "float_dtype"])
def test_public_cpu_loader_checks_complete_headers_before_tensor_unpack(tmp_path, monkeypatch, fault):
    from safetensors.torch import save_file
    c=exact_config()
    (tmp_path/'config.json').write_text(c.to_json_string(use_diff=False))
    tensors={}
    if fault=='extra':tensors['foreign.weight']=torch.zeros(1,dtype=torch.bfloat16)
    elif fault=='shape':tensors['model.language_model.layers.0.self_attn.q_proj.weight_packed']=torch.zeros((1,1),dtype=torch.int32)
    elif fault=='float_dtype':tensors['model.language_model.layers.0.self_attn.q_norm.weight']=torch.ones(128,dtype=torch.float32)
    save_file(tensors,str(tmp_path/'model.safetensors'))
    monkeypatch.setattr(awq,'_pack_projection',lambda *a,**kw:pytest.fail('unpacked incomplete checkpoint'))
    with pytest.raises(ValueError,match='tensor inventory|Unexpected|Invalid'):
        awq.load_qwen3_vl_awq_imagetext(c,str(tmp_path),local_files_only=True)


def test_public_loader_keeps_versioned_cold_tensors_inside_inference_context(tmp_path,monkeypatch):
    c=exact_config()
    (tmp_path/'config.json').write_text(c.to_json_string(use_diff=False))
    class ReachedFactory(Exception):pass
    def factory(config):
        assert not torch.is_inference_mode_enabled()
        assert torch.empty(1)._version == 0
        raise ReachedFactory
    monkeypatch.setattr(awq.AutoModelForImageTextToText,'from_config',factory)
    with torch.inference_mode():
        with pytest.raises(ReachedFactory):
            awq.load_qwen3_vl_awq_imagetext(c,str(tmp_path),local_files_only=True)
        assert torch.is_inference_mode_enabled()
