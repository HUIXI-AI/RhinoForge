"""Run real preprocessing through the complete actual Pi prefill entry.

The complete installed SigLIP caller also executes. Native Vision/Gemma/
planner/KV, Graph execution and Tensor.to(rpu) are host device boundaries.
Their doubles preserve tensor geometry and expose ingress ordering; they are
not a Vision/Gemma math, native Graph, physical SDK or hardware flush oracle.
"""
from pathlib import Path
from types import SimpleNamespace, ModuleType, MethodType
import importlib.util
import ast
from contextlib import contextmanager
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def boundary(monkeypatch):
    monkeypatch.setenv('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
    monkeypatch.delenv('RPU_PI05_PROBE_DIR', raising=False)
    monkeypatch.delenv('RPU_PI05_LOAD_PREFIX_EMBS', raising=False)
    monkeypatch.setenv('RPU_PI05_PREFIX_MASK_CACHE', '1')
    import torch
    # These imports occur in actual preprocessing/prefill. With NCHW256→224,
    # the original interpolation fast path executes; resize's fallback is unused.
    lerobot = ModuleType('lerobot.policies.pi05.modeling_pi05')
    def unused(*args, **kwargs):
        raise AssertionError('unexpected external LeRobot math fallback')
    lerobot.PI05Pytorch = SimpleNamespace(embed_prefix=unused)
    lerobot.resize_with_pad_torch = unused
    lerobot.make_att_2d_masks = unused
    monkeypatch.setitem(sys.modules, 'lerobot.policies.pi05.modeling_pi05', lerobot)

    from rpu_backend.adapters.pi05 import patches
    from rpu_backend.adapters import siglip
    from rpu_backend.api._execution import (normalize_rpu_execution,
        PI05_EXECUTION_SUPPORTED, PI05_EXECUTION_COMPONENTS)

    candidate = _module('_pi05_prefill_candidate156', ROOT / 'python/rpu_backend/adapters/pi05/runtime.py')
    original_to, original_cat = torch.Tensor.to, torch.cat
    uploads, cats = [], []

    class RpuTensor(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type='rpu')

        def cpu(self, *args, **kwargs):
            return self.as_subclass(torch.Tensor).clone()

    def plain(t):
        return t.as_subclass(torch.Tensor) if isinstance(t, RpuTensor) else t

    def raw(t):
        return plain(t).resolve_neg().resolve_conj().detach().contiguous().view(torch.uint8).numpy().tobytes()

    def to(t, *args, **kwargs):
        target = kwargs.get('device', args[0] if args else None)
        kind = getattr(target, 'type', target)
        if kind != 'rpu':
            if kind == 'cpu' and isinstance(t, RpuTensor):
                return original_to(plain(t).clone(), *args, **kwargs)
            return original_to(t, *args, **kwargs)
        dtype = kwargs.get('dtype', t.dtype)
        if isinstance(t, RpuTensor):
            return original_to(t, dtype=dtype)
        converted = original_to(t, dtype=dtype).clone()
        uploaded = converted.as_subclass(RpuTensor)
        uploads.append(dict(shape=tuple(t.shape), dtype=t.dtype, target_dtype=dtype,
                            source=t.data_ptr(), destination=uploaded.data_ptr(), bytes=raw(uploaded)))
        return uploaded

    def cat(tensors, *args, **kwargs):
        tensors = list(tensors)
        if any(isinstance(t, RpuTensor) for t in tensors):
            cats.append([(tuple(t.shape), t.device.type, t.dtype) for t in tensors])
        return original_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', to)
    monkeypatch.setattr(torch, 'cat', cat)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(8)
    handles, vision_callbacks = {}, {}
    monkeypatch.setattr(torch.ops.rpu, 'siglip_get_resolved_chunk_size',
                        lambda handle: 256, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gemma_get_resolved_chunk_size',
                        lambda handle: handles[handle].planned_chunk, raising=False)

    def make_owner(cores, native_handle=137, first_rpu=False, cameras=3, w8=False, uniform=True):
        class Vision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = torch.nn.Linear(3, 3, bias=False)
                self.encoder = torch.nn.Linear(3, 3, bias=False)
                # Actual module registration preserves CPU embeddings first.
                self.encoder.weight = torch.nn.Parameter(
                    self.encoder.weight.detach().as_subclass(RpuTensor), requires_grad=False)
                if first_rpu:
                    self.embeddings.weight = torch.nn.Parameter(
                        self.embeddings.weight.detach().as_subclass(RpuTensor), requires_grad=False)
                self._rpu_vision_handle = native_handle
                self._rpu_vision_num_cores = 8 if cores == 8 else 4
                self._rpu_cache = SimpleNamespace(num_layers=27, num_kv_heads=16, head_dim=80,
                    attn_tp=8 if cores == 8 else 4, max_seq_len=cameras*256,
                    k_caches=[None]*27, v_caches=[None]*27,
                    reset_to_position=lambda position: None)
                self._fmb_last_execution_plan = SimpleNamespace(selected=SimpleNamespace(
                    stage_tuple=SimpleNamespace(input_chunk=4, compute_chunk=256)))

        vision = Vision().eval()
        assert next(vision.parameters()).device.type == ('rpu' if first_rpu else 'cpu')
        assert next(vision.encoder.parameters()).device.type == 'rpu'
        vlm = SimpleNamespace(config=SimpleNamespace(_attn_implementation=''),
                              _rpu_vlm_decoder_handle=id(vision))
        owner = SimpleNamespace(training=False, _rpu_siglip_batch_n=(cameras if cores == 8 else 1),
            _rpu_execution=normalize_rpu_execution({'model': {'num_cores': cores}},
                entry_point='Pi05 complete actual prefill entry host fixture',
                supported=PI05_EXECUTION_SUPPORTED, supported_components=PI05_EXECUTION_COMPONENTS),
            _pi05_prefix_assembly_enabled=False, _pi05_prefill_mask_fp16=True,
            _pi05_precompute_adarms=False, cpu_threads=8, planned_chunk=400,
            leases=[], serial=[], packed=[], kv_records=[], training_fallback_calls=0,
            native_inputs=[], native_outputs=[], signatures=[], pending=[], graph_active=False,
            serial_tensors=[])
        assert type(owner._rpu_execution) is not dict and type(owner._rpu_execution['model']) is not dict
        handles[vlm._rpu_vlm_decoder_handle] = owner

        def image_embedding(image, *, private_cpu=False):
            if private_cpu:
                prepared = image
            else:
                prepared = image.to(dtype=torch.float16, device='rpu')
            owner.leases.append(prepared)
            value = plain(prepared).flatten(1)[:, :256, None].expand(1, 256, 2048).clone()
            return value.as_subclass(RpuTensor)

        @contextmanager
        def capture(signature):
            assert not owner.graph_active
            owner.graph_active=True
            owner.signatures.append(signature)
            try:
                yield
                for write in owner.pending:write()
            finally:
                owner.pending.clear();owner.graph_active=False
        vision._rpu_siglip_graph_cache=SimpleNamespace(capture=capture)

        # Extract the COMPLETE installed caller; only its cold weight-binding,
        # planner, native/Graph and optional HF result class are boundaries.
        tree=ast.parse((ROOT/'python/rpu_backend/adapters/siglip.py').read_text())
        forward=next(node for node in ast.walk(tree)
                     if isinstance(node,ast.FunctionDef) and node.name=='rpu_siglip_forward')
        scope=dict(vars(siglip),_num_images=cameras if cores==8 else 1,
            _attn_tp=8 if cores==8 else 4,_num_layers=27,_num_heads=16,
            _padded_head_dim=80,
            _seq_len=256,_siglip_sig_w8a16=int(w8),_split_siglip_graph=True,
            qkv_w8a8=False,mlp_w8a8=False,
            _siglip_sig_qkv_w8a8=0,_siglip_sig_mlp_w8a8=0,
            num_cores=8 if cores==8 else 4,siglip_w8a16=w8,siglip_w4a16=False,
            rpu_backend=SimpleNamespace(graph=SimpleNamespace(GraphSignature=lambda **kwargs:kwargs)))
        names=('q_w_list','k_w_list','v_w_list','o_w_list','fc1_w_list','fc2_w_list')
        for name in names:scope[name]=[torch.ones(1,dtype=torch.int8 if w8 else torch.float16) for _ in range(27)]
        if not uniform:scope['fc2_w_list'][12]=torch.ones(1,dtype=torch.float16 if w8 else torch.int8)
        cold=next(node for node in ast.walk(tree) if isinstance(node,ast.Assign)
                  and any(isinstance(target,ast.Name) and target.id=='_private_serial_cpu_profile'
                          for target in node.targets))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cold],type_ignores=[])),
                     str(ROOT/'python/rpu_backend/adapters/siglip.py'),'exec'),scope)
        assert scope['_private_serial_cpu_profile']==(cores!=8 and uniform)
        def component_plan(model,handle,length,images,**kwargs):
            assert model is vision and handle==native_handle and length==images*256
            result=SimpleNamespace(selected=SimpleNamespace(stage_tuple=SimpleNamespace(
                physical_descriptor=(11,12),input_chunk=4,compute_chunk=256)),
                graph_key_words=lambda:(4 if cores!=8 else 8,images))
            model._fmb_last_execution_plan=result
            return result
        scope['_siglip_component_execution_plan']=component_plan
        exec(compile(ast.fix_missing_locations(ast.Module(body=[forward],type_ignores=[])),
                     str(ROOT/'python/rpu_backend/adapters/siglip.py'),'exec'),scope)
        vision.forward=MethodType(scope['rpu_siglip_forward'],vision)
        handles[native_handle]=owner

        def embed_image(image):
            owner.serial.append((tuple(image.shape),image.device.type,image.dtype))
            owner.serial_tensors.append(image)
            if native_handle is None:
                owner.leases.append(image)
                return plain(image).flatten(1)[:,:256,None].expand(1,256,2048).half().clone()
            return vision(pixel_values=image).last_hidden_state

        def packed(vision_model, images):
            assert vision_model is vision
            prepared = siglip._prepare_siglip_images(images, vision_model=vision_model)
            owner.packed.append([(tuple(t.shape), t.device.type, t.dtype) for t in prepared])
            return original_cat([image_embedding(t, private_cpu=t.device.type == 'cpu') for t in prepared], dim=1)

        vision_callbacks[id(vision)] = packed
        pawe = SimpleNamespace(paligemma=SimpleNamespace(model=SimpleNamespace(
            vision_tower=vision, language_model=vlm)), embed_image=embed_image,
            embed_language_tokens=lambda tokens: tokens.to(torch.float16)[:, :, None].expand(-1, -1, 2048) * 0.001)
        owner.paligemma_with_expert = pawe
        owner.embed_prefix = MethodType(patches._patched_embed_prefix, owner)
        owner._apply_checkpoint = lambda function, *args, **kwargs: function(*args, **kwargs)
        owner._prepare_attention_masks_4d = lambda mask: torch.where(
            mask[:, None], torch.tensor(0., dtype=torch.float16),
            torch.tensor(torch.finfo(torch.float16).min, dtype=torch.float16))

        def kv_only(**kwargs):
            owner.kv_records.append({k: (tuple(v.shape), v.dtype, raw(v)) for k, v in kwargs.items()})
            return SimpleNamespace(k_caches=[None]*18, v_caches=[None]*18)
        vlm._rpu_prefill_kv_only = kv_only
        return owner

    # LeRobot is optional in host CI. Its training/reference entry is an
    # explicit external consumer boundary here, while the actual adapter must
    # delegate to it. Do not skip this branch when the optional package is absent.
    def training_prefix_boundary(owner, images, image_masks, tokens, token_masks):
        owner.training_fallback_calls += 1
        spans = [owner._apply_checkpoint(owner.paligemma_with_expert.embed_image, image)
                 for image in images]
        language = owner._apply_checkpoint(
            owner.paligemma_with_expert.embed_language_tokens, tokens)
        language = language * (language.shape[-1] ** 0.5)
        prefix = original_cat(spans + [language], dim=1)
        padding = original_cat([mask[:, None].expand(-1, span.shape[1])
                                for mask, span in zip(image_masks, spans, strict=True)]
                               + [token_masks], dim=1)
        attention = torch.zeros(prefix.shape[:2], dtype=torch.bool)
        return prefix, padding, attention

    monkeypatch.setattr(patches, '_orig_embed_prefix', training_prefix_boundary)

    def native_forward(handle,image,keys,values,descriptor):
        owner=handles[handle]
        assert owner.graph_active and len(keys)==len(values)==27 and descriptor==[11,12]
        # A native-copy boundary owns complete bytes before delayed execution.
        private=plain(image).clone()
        owner.leases.append(private)
        owner.native_inputs.append((tuple(image.shape),image.device.type,image.dtype,raw(private)))
        output=torch.empty(1,256,2048,dtype=torch.float16).as_subclass(RpuTensor)
        expected=private.flatten(1)[:,:256,None].expand(1,256,2048)
        owner.pending.append(lambda:output.copy_(expected))
        owner.native_outputs.append(output)
        return output
    monkeypatch.setattr(torch.ops.rpu,'siglip_forward',native_forward,raising=False)
    monkeypatch.setattr(torch.ops.rpu,'spm_alloc_reset_temporary',lambda:None,raising=False)
    outputs=ModuleType('transformers.modeling_outputs')
    outputs.BaseModelOutputWithPooling=lambda **kwargs:SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules,'transformers.modeling_outputs',outputs)

    monkeypatch.setattr(siglip, '_siglip_forward_images',
                        lambda vision, images: vision_callbacks[id(vision)](vision, images))
    for runtime in (candidate,):
        def plan(owner, length):
            assert length in (544, 800)
            owner.planned_chunk = length // 2
            descriptor = ('host-native-plan-boundary', length, owner._rpu_execution['model']['num_cores'])
            result = SimpleNamespace(selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=descriptor)),
                physical_plan_digest='physical-'+str(length), plan_digest='plan-'+str(length),
                domain_digest='domain-'+str(length), optimality='COMPLETE', search_complete=True)
            return length, owner.planned_chunk, result
        monkeypatch.setattr(runtime, '_plan_pi05_prefix_execution', plan)
        monkeypatch.setattr(runtime, '_fixed_plan_receipt',
                            lambda plan, **kwargs: dict(kwargs, authority='HOST_NATIVE_PLAN_BOUNDARY'))

    # Three deterministic NCHW256 CPU cameras; no checkpoint or saved artifact.
    image_keys = ['observation.images.camera'+str(i) for i in range(3)]
    samples = torch.arange(3*256*256, dtype=torch.float32).remainder(251).div(250)
    batch = {key: samples.roll(index*17).reshape(1,3,256,256).clone()
             for index,key in enumerate(image_keys)}

    def preprocess(cameras=3, fp16=True):
        keys = image_keys[:cameras]
        policy = SimpleNamespace(config=SimpleNamespace(image_features=dict.fromkeys(keys), image_resolution=(224,224)))
        result = patches._batched_preprocess_images(policy, batch, torch.device('cpu'), fp16_output=fp16)
        assert result is not None
        images, masks = result
        slab = siglip._shared_siglip_image_slab(images)
        assert slab is not None and tuple(slab.shape) == (cameras,3,224,224)
        assert slab.is_contiguous() and slab.storage_offset() == 0
        assert slab.untyped_storage().nbytes() == slab.numel() * slab.element_size()
        # Independent per-camera CPU oracle uses ordinary interpolation and
        # normalization, rather than the adapter's concatenated in-place path.
        for key,image in zip(keys,images,strict=True):
            reference = torch.nn.functional.interpolate(batch[key], size=(224,224),
                mode='bilinear',align_corners=False).clamp(0.0,1.0)
            reference = reference * 2.0 - 1.0
            if fp16:reference=reference.to(torch.float16)
            assert raw(image) == raw(reference) and tuple(image.shape)==(1,3,224,224)
        return images, masks

    def run(runtime, owner, images, masks):
        uploads.clear();cats.clear()
        tokens = torch.arange(32, dtype=torch.int64)[None]
        token_mask = torch.ones(1,32,dtype=torch.bool)
        cache, pad, length = runtime._prefill_prefix_embs(owner, images, masks, tokens, token_mask, 'rpu')
        assert cache._prefix_len == length == len(images)*256+32
        assert tuple(pad.shape) == (1,length) and pad.dtype == torch.bool
        assert owner.kv_records[-1]['inputs_embeds'][:2] == ((1,length,2048),torch.float16)
        assert owner.kv_records[-1]['attention_mask'][:2] == ((1,1,length,length),torch.float16)
        assert owner.kv_records[-1]['position_ids'][:2] == ((1,length),torch.int64)
        assert '_rpu_planned_prefill_plan' not in vars(owner.paligemma_with_expert.paligemma.model.language_model)
        assert owner._rpu_last_execution_plan['prefill']['logical_len'] == length
        # Native compute is a transparent pixel-row boundary. Independently
        # assemble its expected three camera spans and scaled CPU text rows.
        expected_images=[plain(t).to(torch.float16).flatten(1)[:,:256,None].expand(1,256,2048)
                         for t in images]
        language=tokens.to(torch.float16)[:,:,None].expand(1,32,2048)*0.001
        language=language * (2048 ** 0.5)
        expected=original_cat(expected_images+[language],dim=1)
        record=owner.kv_records[-1]
        assert record['inputs_embeds'][2] == raw(expected)
        assert record['attention_mask'][2] == raw(torch.zeros(1,1,length,length,dtype=torch.float16))
        assert record['position_ids'][2] == raw(torch.arange(length,dtype=torch.int64)[None])
        return list(uploads),list(cats),record

    yield SimpleNamespace(torch=torch, candidate=candidate, make_owner=make_owner,
                          preprocess=preprocess, run=run, raw=raw, plain=plain, siglip=siglip)
    torch.set_num_threads(old_threads)


@pytest.mark.parametrize('cores',[4,6])
@pytest.mark.parametrize('w8',[False,True])
def test_actual_prefill_cpu_embeddings_native_encoder_keeps_private_cpu_images(boundary,cores,w8):
    b=boundary
    images,masks=b.preprocess()
    candidate=b.make_owner(cores,w8=w8)
    current_uploads,cats,current_kv=b.run(b.candidate,candidate,images,masks)
    assert current_uploads == []
    assert candidate.serial == [((1,3,224,224),'cpu',b.torch.float16)]*3
    assert [item[:3] for item in candidate.native_inputs] == candidate.serial
    assert [item[3] for item in candidate.native_inputs] == [b.raw(image) for image in images]
    assert len(candidate.signatures)==3 and all(sig==candidate.signatures[0] for sig in candidate.signatures)
    assert candidate.signatures[0]['shapes']==[256,27,80]
    assert candidate.signatures[0]['dyn_dims'][:4]==[16,4,1,int(w8)]
    assert candidate.signatures[0]['branch_key']==1
    assert cats == [[((1,256,2048),'rpu',b.torch.float16)]*3 + [((1,32,2048),'cpu',b.torch.float16)]]
    assert candidate.cpu_threads == 8 and candidate._pi05_precompute_adarms is False
    saved = [(t,b.raw(t)) for t in candidate.leases]
    slab = images[0]._pi05_batch_slab
    for marker in (0.25,-0.375,0.25):
        slab.fill_(marker)
        new_images=[slab.narrow(0,i,1) for i in range(3)]
        # Keep the real producer's marks on these intact updated views.
        for i,t in enumerate(new_images):
            t._pi05_batch_slab=slab;t._pi05_batch_index=i
        before=len(candidate.leases)
        u,_,_=b.run(b.candidate,candidate,new_images,masks)
        assert u==[] and all(b.raw(t)==v for t,v in saved)
        assert all(sig==candidate.signatures[0] for sig in candidate.signatures)
        assert len(candidate.leases)==before+3
    assert len({t.data_ptr() for t,_ in saved})==3


@pytest.mark.parametrize('cores',[4,6])
@pytest.mark.parametrize('case',['fp32','unmarked','allocation_tail','training','mixed','public_rpu',
                                 'zero_handle','negative_handle','bool_handle','string_handle','no_handle',
                                 'reordered','duplicated','lazyneg','grad','subclass','bad_marker'])
def test_nonadmitted_actual_prefill_retains_original_ingress(boundary,cores,case):
    b=boundary;images,masks=b.preprocess(fp16=case!='fp32')
    native_handle={'zero_handle':0,'negative_handle':-1,'bool_handle':True,
                   'string_handle':'137','no_handle':None}.get(case,137)
    if case=='unmarked':
        for t in images:del t._pi05_batch_slab
    elif case=='allocation_tail':
        slab=images[0]._pi05_batch_slab
        backing=b.torch.empty(slab.numel()+32,dtype=slab.dtype)
        new=backing[:slab.numel()].view_as(slab);new.copy_(slab)
        images=[new.narrow(0,i,1) for i in range(3)]
        for i,t in enumerate(images):t._pi05_batch_slab=new;t._pi05_batch_index=i
    elif case=='lazyneg':
        original=images[0];image=b.torch._neg_view(original)
        image._pi05_batch_slab=original._pi05_batch_slab;image._pi05_batch_index=0
        images[0]=image
    elif case=='grad':images[0].requires_grad_(True)
    elif case=='subclass':
        class PlainTensor(b.torch.Tensor):pass
        original=images[0];image=original.as_subclass(PlainTensor)
        image._pi05_batch_slab=original._pi05_batch_slab;image._pi05_batch_index=0
        images[0]=image
    elif case=='bad_marker':images[0]._pi05_batch_index=True
    elif case=='reordered':images=images[::-1]
    elif case=='duplicated':images=[images[0],images[0],images[2]]
    elif case in ('mixed','public_rpu'):
        images=[t.to('rpu') if (case=='public_rpu' or i==0) else t for i,t in enumerate(images)]
    current=b.make_owner(cores,native_handle=native_handle)
    if case=='training':
        current.training=True
        current.paligemma_with_expert.paligemma.model.vision_tower.train()
    u_new,c_new,k_new=b.run(b.candidate,current,images,masks)
    assert [u['shape'] for u in u_new] == ([(1,3,224,224)]*(2 if case=='mixed' else 0 if case in ('public_rpu','no_handle') else 3))
    assert current.training_fallback_calls == (1 if case=='training' else 0)
    if case in ('lazyneg','grad','subclass','bad_marker'):
        assert current.serial_tensors[0] is images[0]
        assert hasattr(images[0],'_pi05_batch_slab') and hasattr(images[0],'_pi05_batch_index')
        assert [item[1] for item in current.native_inputs]==['rpu']*3
    if case in ('mixed','reordered','duplicated'):
        assert [item[1] for item in current.native_inputs]==['rpu']*3
        for original,local in zip(images,current.serial_tensors,strict=True):
            if type(original) is b.torch.Tensor and original.device.type=='cpu':
                assert local is not original
                assert not hasattr(local,'_pi05_batch_slab') and not hasattr(local,'_pi05_batch_index')
                assert hasattr(original,'_pi05_batch_slab') and hasattr(original,'_pi05_batch_index')
                assert tuple(local.shape)==tuple(original.shape) and tuple(local.stride())==tuple(original.stride())
                assert local.dtype==original.dtype and local.storage_offset()==original.storage_offset()
                assert local.data_ptr()==original.data_ptr()
                assert local.untyped_storage()._cdata==original.untyped_storage()._cdata
                assert b.raw(local)==b.raw(original)
            else:assert local is original



@pytest.mark.parametrize('cores,cameras,first_rpu',[(8,2,False),(8,3,False),(4,3,True),(6,3,True)])
def test_packed_TP8_and_original_first_rpu_route_are_unchanged(boundary,cores,cameras,first_rpu):
    b=boundary;images,masks=b.preprocess(cameras)
    current=b.make_owner(cores,first_rpu=first_rpu,cameras=cameras)
    u_new,c_new,k_new=b.run(b.candidate,current,images,masks)
    if cores==8:
        # This host boundary explicitly selects the existing reference prefix
        # composition, so both versions upload its final prefix after CPU cat.
        assert [u["shape"] for u in u_new] == [(1,cameras*256+32,2048)]
        assert len(current.packed)==1
    else:
        assert u_new==[]
        assert [item[1] for item in current.native_inputs]==['cpu']*3


@pytest.mark.parametrize('cores',[4,6])
def test_partial_precision_cold_profile_keeps_original_single_image_upload(boundary,cores):
    b=boundary;images,masks=b.preprocess()
    owner=b.make_owner(cores,w8=True,uniform=False)
    uploads,_,_=b.run(b.candidate,owner,images,masks)
    assert [upload['shape'] for upload in uploads]==[(1,3,224,224)]*3
    assert [item[1] for item in owner.native_inputs]==['rpu']*3


def test_masked_language_rows_reach_prefill_as_zero(boundary):
    """With the H10 weights a pad row's own residuals overflow FP16 at layer 16."""
    from rpu_backend.adapters.pi05 import patches
    torch=boundary.torch
    tokens=torch.tensor([[2,5,0,0]],dtype=torch.int64)
    masks=tokens!=0
    owner=SimpleNamespace(paligemma_with_expert=SimpleNamespace(
        embed_language_tokens=lambda t:(t+1).to(torch.float16)[:,:,None].expand(-1,-1,4)))
    full=patches._embed_language_rows(owner,tokens,torch.ones_like(masks))
    padded=patches._embed_language_rows(owner,tokens,masks)
    assert full[0,:,0].tolist()==[6,12,2,2]
    assert torch.equal(padded[:,:2],full[:,:2])
    assert not padded[:,2:].any()
