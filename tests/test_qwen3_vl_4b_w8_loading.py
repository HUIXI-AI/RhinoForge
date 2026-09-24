"""Host coverage of the controlled 4B W8 storage/shape boundary, not board certification."""
import copy
import json
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import save_file

from rpu_backend.quant import load
from rpu_backend.quant import convert_qwen3_vl as convert
from rpu_backend.quant._common import quantize_linear_per_channel
from rpu_backend.adapters import qwen3_vl as adapter
from rpu_backend.adapters.qwen3_vl import text


def config():
    return NS(
        architectures=["Qwen3VLForConditionalGeneration"], model_type="qwen3_vl",
        tie_word_embeddings=False, quant_config=copy.deepcopy(convert.QUANT_CONFIG),
        image_token_id=151655, video_token_id=151656,
        vision_start_token_id=151652, vision_end_token_id=151653,
        text_config=NS(model_type="qwen3_vl_text", hidden_size=2560,
            intermediate_size=9728, num_hidden_layers=36, num_attention_heads=32,
            num_key_value_heads=8, head_dim=128, vocab_size=151936,
            hidden_act="silu", rms_norm_eps=1e-6, attention_bias=False,
            use_cache=True, tie_word_embeddings=False, max_position_embeddings=262144,
            rope_parameters={"rope_type":"default", "rope_theta":5000000.0,
                             "mrope_interleaved":True,"mrope_section":[24,20,20]}),
        vision_config=NS(model_type="qwen3_vl", hidden_size=1024,
            intermediate_size=4096, depth=24, num_heads=16, patch_size=16,
            temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=2560,
            hidden_act="gelu_pytorch_tanh", in_channels=3, num_position_embeddings=2304,
            deepstack_visual_indexes=[5,11,17]),
    )


def test_exact_admission_without_32b_optin(monkeypatch):
    monkeypatch.delenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED", raising=False)
    cfg = config()
    assert load.is_qwen3_vl_4b_w8a16_config(cfg)
    assert not load.is_qwen3_vl_32b_w8a16_config(cfg)
    adapter.Qwen3VLAdapter.preflight(cfg)


@pytest.mark.parametrize("fault", ["scope", "head", "tied", "rope", "geometry", "nested"])
def test_profile_cannot_silently_widen(fault):
    cfg = config()
    if fault == "scope": cfg.quant_config["vision_quantization"] = "fp16"
    elif fault == "head": cfg.quant_config["quantized_lm_head"] = False
    elif fault == "tied": cfg.text_config.tie_word_embeddings = True
    elif fault == "rope": cfg.text_config.rope_parameters["mrope_section"] = [20,20,24]
    elif fault == "geometry": cfg.vision_config.out_hidden_size = 2048
    else: cfg.vision_config.quantization_config = {"bits":4}
    assert not load.is_qwen3_vl_4b_w8a16_config(cfg)
    with pytest.raises(adapter.UnsupportedModelError):
        adapter.Qwen3VLAdapter.preflight(cfg)


def test_bounded_quantizer_matches_original_rounding_and_zero_rows():
    torch.manual_seed(42)
    weight = torch.randn(4101, 32).to(torch.bfloat16)
    weight[0].zero_()
    expected = quantize_linear_per_channel(weight)
    actual = convert.quantize_weight(weight)
    assert all(torch.equal(a,b) for a,b in zip(expected,actual,strict=True))
    assert actual[0].dtype == torch.int8 and actual[1].dtype == torch.float16
    assert bool(torch.isfinite(actual[1]).all()) and bool((actual[1] > 0).all())


class Tiny(torch.nn.Module):
    """Injected model factory limits data volume while exercising the real loader."""
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.embed_tokens = torch.nn.Embedding(4, 2)
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(2, 3, bias=False)
        self.model.language_model.layers = torch.nn.ModuleList([layer])
        self.model.visual = torch.nn.Module()
        self.model.visual.proj = torch.nn.Linear(2, 2, bias=False)
        self.lm_head = torch.nn.Linear(2, 4, bias=False)


def tensors():
    return {
        "model.language_model.embed_tokens.weight": torch.arange(8).reshape(4,2).half(),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(3,2,dtype=torch.int8),
        "model.language_model.layers.0.self_attn.q_proj.weight_scale": torch.ones(3,dtype=torch.float16),
        "model.visual.proj.weight": torch.ones(2,2,dtype=torch.float16),
        "lm_head.weight": torch.ones(4,2,dtype=torch.int8),
        "lm_head.weight_scale": torch.ones(4,dtype=torch.float16)*.5,
    }


def test_loader_preserves_head_int8_and_raw_embedding(monkeypatch,tmp_path):
    monkeypatch.setattr(load.AutoModelForImageTextToText,"from_config",Tiny)
    save_file(tensors(),tmp_path/"model.safetensors")
    model=load.load_w8a16_imagetext(config(),str(tmp_path))
    assert model.lm_head.weight.dtype == torch.int8
    assert torch.equal(model.lm_head.weight_scale, torch.full((4,),.5,dtype=torch.float16))
    emb=model.model.language_model.embed_tokens.weight
    assert emb.dtype == torch.float16
    assert emb.data_ptr() != model.lm_head.weight.data_ptr()
    assert torch.equal(emb,torch.arange(8).reshape(4,2).half())
    assert model.model.visual.proj.weight.dtype == torch.float16


@pytest.mark.parametrize("fault",["floating_head","missing_scale","wrong_scale_shape","nan_scale","zero_scale"])
def test_bad_head_rejected_before_use(monkeypatch,tmp_path,fault):
    monkeypatch.setattr(load.AutoModelForImageTextToText,"from_config",Tiny)
    data=tensors()
    if fault == "floating_head": data["lm_head.weight"]=data["lm_head.weight"].half()
    elif fault == "missing_scale": del data["lm_head.weight_scale"]
    elif fault == "wrong_scale_shape": data["lm_head.weight_scale"]=torch.ones(3,dtype=torch.float16)
    elif fault == "nan_scale": data["lm_head.weight_scale"][0]=float("nan")
    else: data["lm_head.weight_scale"][0]=0
    save_file(data,tmp_path/"model.safetensors")
    with pytest.raises((ValueError,KeyError,TypeError)):
        load.load_w8a16_imagetext(config(),str(tmp_path))


def test_staged_plan_includes_head_without_materializing(monkeypatch,tmp_path):
    monkeypatch.setattr(load.AutoModelForImageTextToText,"from_config",Tiny)
    save_file(tensors(),tmp_path/"model.safetensors")
    model=load.stage_w8a16_imagetext_for_rpu(config(),str(tmp_path))
    plan=load._validate_staged_w8a16_imagetext(model)
    assert all(p.is_meta for p in model.parameters())
    assert "lm_head.weight_scale" in plan.nondecoder_tensor_names
    assert "lm_head.weight" in load._expected_imagetext_quant_weights(model)


def test_4b_long_envelope_cold_binding_preserves_other_profiles(monkeypatch):
    captured={}
    def install(model,**kwargs):
        captured.update(kwargs)
        return 17
    monkeypatch.setattr(text,"install_mrope_text_decoder_for_rpu",install)
    cfg=config().text_config
    model=NS(config=cfg)
    text.install_qwen3_vl_text_for_rpu(model,scale_lists=([],)*7,runtime_align_w8a16=True)
    actual=captured["chunk_envelope_for"]("qwen3_vl_text",36,2560)
    assert actual == text.ChunkEnvelope(4176,256)
    text.install_qwen3_vl_text_for_rpu(model)
    assert captured["chunk_envelope_for"]("qwen3_vl_text",36,2560) == text.ChunkEnvelope(4176,256)
    assert text.lookup_causal_decoder("qwen3_vl_text",28,2048) == text.ChunkEnvelope(4176,320)
    assert text.lookup_causal_decoder("qwen3_vl_text",36,4096) == text.ChunkEnvelope(4176,128)
    assert text.lookup_causal_decoder("qwen3_vl_text",16,2048) == text.ChunkEnvelope(512)
    with pytest.raises(RuntimeError,match="no certified chunk envelope"):
        text.lookup_causal_decoder("qwen3_vl_text",36,2561)
    with pytest.raises(ValueError,match="scales"):
        text.install_qwen3_vl_text_for_rpu(model,runtime_align_w8a16=True)


@pytest.mark.parametrize("selection",[slice(None),slice(-1,None),torch.tensor([0,2])])
def test_prefill_head_keeps_requested_rows(monkeypatch,selection):
    head=torch.nn.Linear(2,4,bias=False)
    head.weight=torch.nn.Parameter(torch.arange(8,dtype=torch.int8).reshape(4,2),requires_grad=False)
    head.register_buffer("weight_scale",torch.full((4,),.125,dtype=torch.float16))
    hidden=torch.arange(8,dtype=torch.float16).reshape(1,4,2)
    calls=[]
    def native(x,w,bias,partition,cores,acc32,s):
        assert (partition, cores, acc32) == (1, 8, True)
        calls.append((x,w,s))
        return torch.nn.functional.linear(x,(w.float()*s[:,None]).half(),bias)
    monkeypatch.setattr(torch.ops.rpu,"linear_with_accumulation",native,raising=False)
    model=NS(lm_head=head,model=NS(language_model=NS(_rpu_execution={"prefill":{"linear_acc32":True}})))
    actual=adapter._apply_prefill_lm_head(model,hidden,selection)
    expected=torch.nn.functional.linear(hidden[:,selection,:],head.weight.half()*head.weight_scale[:,None])
    assert torch.equal(actual,expected)
    assert calls[0][1] is head.weight and calls[0][2] is head.weight_scale and calls[0][0].is_contiguous()


def test_converter_rejects_quantized_or_incomplete_source(tmp_path):
    source=tmp_path/"src"
    source.mkdir()
    cfg=config()
    raw=json.loads(json.dumps(cfg,default=vars))
    (source/"config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError,match="unquantized"):
        convert.convert_checkpoint(source,tmp_path/"output")
    assert not (tmp_path/"output").exists()


@pytest.mark.parametrize("load_mode", ["staged", "cpu"])
def test_4b_projection_storage_preserves_swizzled_views(monkeypatch,tmp_path,load_mode):
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
    data["lm_head.weight"]=torch.ones((16,256),dtype=torch.int8)
    data["lm_head.weight_scale"]=torch.ones(16,dtype=torch.float16)
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
          else load._move_materialized_w8a16_decoder_for_rpu)
    move(model)
    assert allocations==[14*256*256]
    assert moved==list(model.model.language_model.layers)
    parameters={name:model.get_parameter(name) for name in expected}
    assert len({p.untyped_storage().data_ptr() for p in parameters.values()})==1
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
        spans.append((parameter.storage_offset(),parameter.storage_offset()+parameter.numel()))
    spans.sort()
    assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
    # Module Parameters own the storage after the materializer's local bank
    # and view dictionary have gone out of scope; changing one cannot alter another.
    first_name=next(iter(parameters))
    parameters[first_name].view(-1)[0]=0
    assert all(torch.equal(parameter,expected[name]) for name,parameter in parameters.items() if name!=first_name)
    assert torch.equal(model.model.language_model.embed_tokens.weight,data["model.language_model.embed_tokens.weight"])
    assert model.lm_head.weight.dtype==torch.int8
    assert torch.equal(model.lm_head.weight_scale,data["lm_head.weight_scale"])
    for index,layer in enumerate(model.model.language_model.layers):
        for role in roles:
            assert torch.equal(layer.get_submodule(role).weight_scale,
                               data[f"model.language_model.layers.{index}.{role}.weight_scale"])
    assert load._W8A16_IMAGETEXT_STAGE_ATTR not in vars(model)
    if load_mode=="cpu":
        with pytest.raises(ValueError,match="unswizzled CPU INT8"):
            move(model)
        assert allocations==[14*256*256]


def test_exact_4b_projection_bank_offsets_beyond_two_gib(monkeypatch):
    """Exercise the real cold layout at full geometry without materializing GBs."""
    shapes = {"self_attn.q_proj": (4096, 2560),
              "self_attn.k_proj": (1024, 2560),
              "self_attn.v_proj": (1024, 2560),
              "self_attn.o_proj": (2560, 4096),
              "mlp.gate_proj": (9728, 2560),
              "mlp.up_proj": (9728, 2560),
              "mlp.down_proj": (2560, 9728)}
    raw = {f"model.language_model.layers.{i}.{role}.weight":
           torch.empty(shape, dtype=torch.int8, device="meta")
           for i in range(36) for role, shape in shapes.items()}
    names = tuple(tuple(sorted(n for n in raw if n.startswith(f"model.language_model.layers.{i}.")))
                  for i in range(36))
    allocations = []
    original = torch.empty
    def meta_empty(*args, **kwargs):
        assert kwargs["device"] == "rpu"
        allocations.append(args[0])
        return original(*args, **{**kwargs, "device": "meta"})
    monkeypatch.setattr(torch, "empty", meta_empty)
    views = load._allocate_w8a16_decoder_projection_views(NS(get_parameter=raw.__getitem__), names)
    assert allocations == [sum(w.numel() for w in raw.values())]
    assert len({w.untyped_storage()._cdata for w in views.values()}) == 1
    assert max(w.storage_offset() for w in views.values()) > 2**31
    for i in range(36):
        q, k, v = [views[f"model.language_model.layers.{i}.self_attn.{r}_proj.weight"]
                   for r in ("q", "k", "v")]
        assert q.storage_offset() % 256 == 0
        assert k.storage_offset() == q.storage_offset() + q.numel()
        assert v.storage_offset() == k.storage_offset() + k.numel()
        packed = q.as_strided((6144, 2560), (2560, 1), q.storage_offset())
        assert packed.storage_offset() == q.storage_offset()
        assert packed.untyped_storage()._cdata == q.untyped_storage()._cdata


def test_staged_32b_keeps_original_per_layer_transfer(monkeypatch,tmp_path):
    cfg = config()
    cfg.quant_config = dict(load._QWEN3_VL_32B_W8A16_QUANT_CONFIG)
    vars(cfg.text_config).update(hidden_size=5120, intermediate_size=25600,
        num_hidden_layers=64, num_attention_heads=64)
    vars(cfg.vision_config).update(hidden_size=1152, intermediate_size=4304,
        depth=27, out_hidden_size=5120, deepstack_visual_indexes=[8, 16, 24])
    assert load.is_qwen3_vl_32b_w8a16_config(cfg)
    monkeypatch.setattr(load.AutoModelForImageTextToText, "from_config", Tiny)
    data = tensors()
    data["lm_head.weight"] = data["lm_head.weight"].half()
    del data["lm_head.weight_scale"]
    save_file(data, tmp_path / "model.safetensors")
    model = load.stage_w8a16_imagetext_for_rpu(cfg, str(tmp_path))
    def forbidden(*args,**kwargs):
        raise AssertionError("32B must not allocate the controlled 4B projection bank")
    monkeypatch.setattr(load,"_allocate_w8a16_decoder_projection_views",forbidden)
    calls=[]
    monkeypatch.setattr(load,"_swizzle_and_move_w8a16_decoder_layer",lambda layer:calls.append(layer))
    vars(model)["_rpu_swizzle_started"]=True
    load._materialize_staged_w8a16_imagetext_for_rpu(model)
    assert calls==list(model.model.language_model.layers)


@pytest.mark.parametrize("selection", [slice(None), slice(-1, None), torch.tensor([0, 2])])
def test_prefill_packed_w4_head_keeps_requested_rows_and_physical_owners(monkeypatch, selection):
    from rpu_backend.quant.int4_pgrp_pack import (
        quantize_int4_group_wise, dequantize_int4_group_wise,
        swizzle_pack_int4_pgrp, swizzle_int4_pgrp_scale,
    )
    generator = torch.Generator().manual_seed(920)
    raw = torch.randn(128, 64, generator=generator).half()
    values, logical_scale, _ = quantize_int4_group_wise(raw, 32)
    head = torch.nn.Linear(64, 128, bias=False)
    head.weight = torch.nn.Parameter(
        swizzle_pack_int4_pgrp(values, 1, 8).reshape(128, 32), requires_grad=False)
    head.register_buffer("weight_scale", swizzle_int4_pgrp_scale(logical_scale, 32, 1, 8))
    hidden = torch.randn(1, 4, 64, generator=generator).half()
    dequantized = dequantize_int4_group_wise(values, logical_scale)
    calls = []
    def native(x, weight, bias, partition, cores, acc32, scale):
        assert (partition, cores, acc32) == (1, 8, False)
        calls.append((x, weight, scale))
        return torch.nn.functional.linear(x, dequantized, bias)
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", native, raising=False)
    monkeypatch.setattr(head, "forward", lambda *a, **k: pytest.fail("packed head entered float Linear"))
    model = NS(lm_head=head, model=NS(language_model=NS(_rpu_execution={})))
    output = adapter._apply_prefill_lm_head(model, hidden, selection)
    assert torch.equal(output, torch.nn.functional.linear(hidden[:, selection, :], dequantized))
    assert len(calls) == 1 and calls[0][0].is_contiguous()
    assert calls[0][1] is head.weight and calls[0][2] is head.weight_scale
