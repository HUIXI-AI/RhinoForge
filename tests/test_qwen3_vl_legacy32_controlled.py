import copy
from types import SimpleNamespace as NS
import pytest
import torch
from safetensors.torch import save_file
from rpu_backend.quant import load
from rpu_backend.adapters import qwen3_vl as adapter
from rpu_backend.adapters.qwen3_vl import text
from test_qwen3_vl_4b_w8_loading import config as config4

def config():
    cfg=config4()
    cfg.quant_config=copy.deepcopy(load._QWEN3_VL_32B_W8A16_QUANT_CONFIG)
    vars(cfg.text_config).update(hidden_size=5120,intermediate_size=25600,num_hidden_layers=64,num_attention_heads=64)
    vars(cfg.vision_config).update(hidden_size=1152,intermediate_size=4304,depth=27,out_hidden_size=5120,deepstack_visual_indexes=[8,16,24])
    assert load.is_qwen3_vl_32b_w8a16_config(cfg)
    return cfg

@pytest.fixture(autouse=True)
def bounded_threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)

@pytest.mark.parametrize("load_mode", ["staged", "cpu"])
def test_legacy_projection_banks_preserve_payload_and_owners(monkeypatch,tmp_path,load_mode):
    from rpu_backend.runtime import weights

    roles = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

    class PackedTiny(torch.nn.Module):
        def __init__(self,cfg):
            super().__init__()
            self.config=cfg
            self.model=torch.nn.Module()
            self.model.language_model=torch.nn.Module()
            self.model.language_model.embed_tokens=torch.nn.Embedding(16,256)
            self.model.language_model.layers=torch.nn.ModuleList()
            for _ in range(2):
                layer=torch.nn.Module()
                layer.self_attn=torch.nn.Module()
                layer.mlp=torch.nn.Module()
                for role in roles:
                    owner,name=role.split(".")
                    getattr(layer,owner).add_module(name,torch.nn.Linear(256,256,bias=False))
                layer.norm=torch.nn.LayerNorm(256,bias=False)
                self.model.language_model.layers.append(layer)
            self.model.visual=torch.nn.Module()
            self.model.visual.proj=torch.nn.Linear(256,256,bias=False)
            self.lm_head=torch.nn.Linear(256,16,bias=False)

    skeleton=PackedTiny(config())
    generator=torch.Generator().manual_seed(916)
    data={name:parameter.detach().half() for name,parameter in skeleton.named_parameters()}
    expected={}
    for index in range(2):
        for role in roles:
            name=f"model.language_model.layers.{index}.{role}.weight"
            raw=torch.randint(-127,128,(256,256),dtype=torch.int8,generator=generator)
            data[name]=raw
            data[name[:-len(".weight")]+".weight_scale"]=torch.full((256,),.125,dtype=torch.float16)
            partition=0 if role.endswith(("o_proj","down_proj")) else 1
            expected[name]=weights.transform_linear_weight(raw,partition)
    data["lm_head.weight"]=torch.ones((16,256),dtype=torch.float16)
    save_file(data,tmp_path/"model.safetensors")
    monkeypatch.setattr(load.AutoModelForImageTextToText,"from_config",PackedTiny)
    loader=(load.stage_w8a16_imagetext_for_rpu if load_mode=="staged"
            else load.load_w8a16_imagetext)
    model=loader(config(),str(tmp_path))

    # Replace only transport: the real strict metadata reader, tensor loader,
    # swizzler and Parameter/view installation run on a bounded CPU fixture.
    real_empty=torch.empty
    allocations=[]
    def host_empty(*args,**kwargs):
        if kwargs.get("device")=="rpu":
            assert kwargs["dtype"]==torch.int8
            allocations.append(args[0])
            kwargs["device"]="cpu"
        return real_empty(*args,**kwargs)
    monkeypatch.setattr(torch,"empty",host_empty)
    moved=[]
    def host_move(module,device):
        assert device=="rpu"
        moved.append(module)
        assert len({module.get_submodule(role).weight.untyped_storage().data_ptr() for role in roles})==1
        return module
    monkeypatch.setattr(torch.nn.Module,"to",host_move)
    vars(model)["_rpu_swizzle_started"]=True
    move=(load._materialize_staged_w8a16_imagetext_for_rpu if load_mode=="staged"
          else lambda m: load._move_materialized_decoder_for_rpu(m,dtype=torch.int8,per_layer=True))
    move(model)
    assert allocations==[7*256*256]*2
    assert moved==list(model.model.language_model.layers)
    parameters={name:model.get_parameter(name) for name in expected}
    assert len({p.untyped_storage().data_ptr() for p in parameters.values()})==2
    for index in range(2):
        q, k, v = [parameters[f"model.language_model.layers.{index}.self_attn.{role}_proj.weight"]
                   for role in ("q", "k", "v")]
        assert k.storage_offset() == q.storage_offset() + q.numel()
        assert v.storage_offset() == k.storage_offset() + k.numel()
        packed = q.as_strided((768, 256), (256, 1), q.storage_offset())
        assert packed.untyped_storage().data_ptr() == q.untyped_storage().data_ptr()
        assert torch.equal(packed, torch.cat((q, k, v)))
    spans=[]
    for name,parameter in parameters.items():
        assert parameter.dtype==torch.int8 and parameter.is_contiguous()
        assert parameter.shape==expected[name].shape and not parameter.requires_grad
        assert parameter.storage_offset()%256==0
        assert torch.equal(parameter,expected[name])
        spans.append((parameter.untyped_storage().data_ptr(),parameter.storage_offset(),parameter.storage_offset()+parameter.numel()))
    spans.sort()
    assert all(a[0]!=b[0] or a[2]<=b[1] for a,b in zip(spans,spans[1:]))
    # Module Parameters own the storage after the materializer's local bank
    # and view dictionary have gone out of scope; changing one cannot alter another.
    first_name=next(iter(parameters))
    parameters[first_name].view(-1)[0]=0
    assert all(torch.equal(parameter,expected[name]) for name,parameter in parameters.items() if name!=first_name)
    assert torch.equal(model.model.language_model.embed_tokens.weight,data["model.language_model.embed_tokens.weight"])
    assert model.lm_head.weight.dtype==torch.float16
    assert torch.equal(model.lm_head.weight,data["lm_head.weight"])
    assert model.model.visual.proj.weight.dtype==torch.float16
    for index,layer in enumerate(model.model.language_model.layers):
        for role in roles:
            assert torch.equal(layer.get_submodule(role).weight_scale,
                               data[f"model.language_model.layers.{index}.{role}.weight_scale"])
    assert load._W8A16_IMAGETEXT_STAGE_ATTR not in vars(model)
    if load_mode=="cpu":
        with pytest.raises(ValueError,match="unswizzled CPU INT8"):
            move(model)
        assert allocations==[7*256*256]*2


@pytest.mark.parametrize("chunk",["auto",16,64])
def test_exact_preflight_has_own_narrow_envelope(monkeypatch,chunk):
    monkeypatch.setenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED","1")
    adapter.Qwen3VLAdapter.preflight(config())
    adapter.Qwen3VLAdapter.preflight_execution(config(),{"prefill":{"chunk_size":chunk}})
    assert text._LEGACY_32B_CONTROLLED_ENVELOPE==(336,64)
    assert text._RUNTIME_QUANTIZED_32B_ENVELOPE==(4176,64)

@pytest.mark.parametrize("chunk",[80,128,256])
def test_wrong_chunk_early_reject(monkeypatch,chunk):
    monkeypatch.setenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED","1")
    with pytest.raises(adapter.UnsupportedModelError,match="ceiling 64"):
        adapter.Qwen3VLAdapter.preflight_execution(config(),{"prefill":{"chunk_size":chunk}})

@pytest.mark.parametrize("value",[None,"true","0"])
def test_unchanged_optin_required(monkeypatch,value):
    if value is None:monkeypatch.delenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED",raising=False)
    else:monkeypatch.setenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED",value)
    with pytest.raises(adapter.UnsupportedModelError):adapter.Qwen3VLAdapter.preflight(config())

def meta_text(dtype=torch.int8):
    cfg=config().text_config;m=torch.nn.Module();m.config=cfg
    shapes=(("self_attn.q_proj",8192,5120),("self_attn.k_proj",1024,5120),("self_attn.v_proj",1024,5120),("self_attn.o_proj",5120,8192),("mlp.gate_proj",25600,5120),("mlp.up_proj",25600,5120),("mlp.down_proj",5120,25600))
    m.layers=torch.nn.ModuleList();scales=[[] for _ in range(7)]
    for i in range(64):
        layer=torch.nn.Module();layer.self_attn=torch.nn.Module();layer.mlp=torch.nn.Module()
        for j,(role,n,k) in enumerate(shapes):
            p=torch.nn.Linear(k,n,bias=False,device="meta");p.weight=torch.nn.Parameter(torch.empty(n,k//2 if dtype==torch.uint8 else k,dtype=dtype,device="meta"),requires_grad=False)
            p.register_buffer("weight_scale",torch.empty((32,n*k//1024) if dtype==torch.uint8 else (n,),dtype=torch.float16,device="meta"));scales[j].append(p.weight_scale)
            owner,name=role.split(".");getattr(layer,owner).add_module(name,p)
        m.layers.append(layer)
    m.embed_tokens=torch.nn.Embedding(151936,5120,dtype=torch.float16,device="meta")
    return m,scales

def test_exact_live_binding_and_independent_envelope(monkeypatch):
    monkeypatch.setenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED","1")
    m,scales=meta_text();seen={}
    monkeypatch.setattr(text,"install_mrope_text_decoder_for_rpu",lambda m,**kw:seen.update(kw) or 1)
    assert text.install_qwen3_vl_text_for_rpu(m,scale_lists=scales,_legacy_32b_w8a16=True)==1
    assert seen['chunk_envelope_for']('qwen3_vl_text',64,5120)==(336,64)
    assert seen['_kv_cache_layer_bank_size']==8
    assert '_runtime_quantized_32b' not in seen

@pytest.mark.parametrize("fault",['w4','scale_alias','scale_dtype','geometry','embed','both_profiles','tp4'])
def test_reject_changed_live_binding_before_installer(monkeypatch,fault):
    monkeypatch.setenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED","1")
    m,scales=meta_text(torch.uint8 if fault=='w4' else torch.int8);kw={}
    if fault=='scale_alias':scales[0][0]=scales[0][0].clone()
    elif fault=='scale_dtype':m.layers[0].self_attn.q_proj.weight_scale=scales[0][0]=scales[0][0].float()
    elif fault=='geometry':m.config.hidden_size=4096
    elif fault=='embed':m.embed_tokens.weight=torch.nn.Parameter(m.embed_tokens.weight.float())
    elif fault=='both_profiles':kw['_runtime_quantized_32b']=True
    elif fault=='tp4':kw['execution_config']={'model':{'num_cores':4}}
    monkeypatch.setattr(text,'install_mrope_text_decoder_for_rpu',lambda *a,**kw:pytest.fail('invalid profile reached native installer'))
    with pytest.raises(ValueError):text.install_qwen3_vl_text_for_rpu(m,scale_lists=scales,_legacy_32b_w8a16=True,**kw)

def test_exact_legacy_bank_allocations_below_four_gib(monkeypatch):
    shapes={'self_attn.q_proj':(8192,5120),'self_attn.k_proj':(1024,5120),'self_attn.v_proj':(1024,5120),'self_attn.o_proj':(5120,8192),'mlp.gate_proj':(25600,5120),'mlp.up_proj':(25600,5120),'mlp.down_proj':(5120,25600)}
    raw={f'model.language_model.layers.0.{r}.weight':torch.empty(sh,dtype=torch.int8,device='meta') for r,sh in shapes.items()};alloc=[];original=torch.empty
    def empty(*a,**k):
        assert k['device']=='rpu';alloc.append(a[0]);k['device']='meta';return original(*a,**k)
    monkeypatch.setattr(torch,'empty',empty)
    views=load._allocate_decoder_projection_views(NS(get_parameter=raw.__getitem__),(tuple(raw),),dtype=torch.int8)
    assert alloc==[487587840] and alloc[0]<2**32
    assert len({v.untyped_storage()._cdata for v in views.values()})==1

from rpu_backend.tests.acceptance.test_qwen3_vl_install_lifecycle import install_runtime, _Model

@pytest.mark.parametrize('failure',['claim','arena','post_poison'])
def test_legacy_cold_arena_ownership_failure_boundary(install_runtime,monkeypatch,failure):
    import rpu_backend
    monkeypatch.setenv('QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED','1')
    model=_Model();model.config=config();state=install_runtime;released=[]
    # Tiny tensors test the real top-level lifecycle; exact shape/schema
    # admission is independently covered above and by the strict loader.
    monkeypatch.setattr(adapter,'_validate_qwen3_vl_32b_w8a16_model',lambda m:True)
    monkeypatch.setattr(adapter,'_release_live_instance',lambda m:released.append(m))
    def arenas(*,graph_arena_count):
        assert graph_arena_count==25
        assert state['claims']==[model] and state['allocator_enabled']
        assert '_rpu_swizzle_started' not in vars(model)
        assert not any(e[0] in ('swizzle','rpu_allocation','install') for e in state['events'])
        state['events'].append(('arenas',25))
        return NS(prepare_arenas=lambda:failure!='arena')
    monkeypatch.setattr(rpu_backend.graph.GraphRuntimePolicy,'from_environment',arenas)
    def move(m,*,dtype,per_layer):
        assert m is model and dtype==torch.int8 and per_layer is True
        assert ('arenas',25) in state['events'] and vars(model)['_rpu_swizzle_started'] is True
        raise RuntimeError('bounded transport failure')
    monkeypatch.setattr(load,'_move_materialized_decoder_for_rpu',move)
    if failure=='claim':state['claim_error']=RuntimeError('active owner')
    a=adapter.Qwen3VLAdapter(model)
    with pytest.raises((RuntimeError,adapter.RPUBackendError)):a.to_rpu()
    assert not a._rpu_is_ready
    if failure=='claim':
        assert state['events']==[] and released==[]
        assert '_rpu_swizzle_started' not in vars(model)
    elif failure=='arena':
        assert released==[model] and '_rpu_swizzle_started' not in vars(model)
        assert state['events']==[('claim','model'),('set_caching_allocator',True),('arenas',25)]
    else:
        assert vars(model).get('_rpu_swizzle_started') is True
        assert a._closed
        with pytest.raises((RuntimeError,adapter.RPUBackendError)):a.to_rpu()
