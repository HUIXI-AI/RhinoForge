"""Actual six-projection quantization/swizzle on CPU; no native/board required."""
from copy import deepcopy

import pytest
import torch

from rpu_backend.adapters.qwen3_5 import vision
from rpu_backend.quant._common import quantize_linear_per_channel
from rpu_backend.runtime.weights import tp_col_swizzle_mc_weight, tp_row_swizzle_mc_weight


def block():
    b = torch.nn.Module()
    b.attn = torch.nn.Module()
    b.attn.qkv = torch.nn.Linear(1024,3072,dtype=torch.float16)
    b.attn.proj = torch.nn.Linear(1024,1024,dtype=torch.float16)
    b.mlp = torch.nn.Module()
    b.mlp.linear_fc1 = torch.nn.Linear(1024,4096,dtype=torch.float16)
    b.mlp.linear_fc2 = torch.nn.Linear(4096,1024,dtype=torch.float16)
    return b


def test_vision_w8_six_real_projection_scales_swizzle_and_fp16_default():
    torch.manual_seed(138)
    b = block()
    original = deepcopy(b)
    vision._convert_vision_block_weights_for_rpu(b,16,1024,_g05_w8a16=True)
    raw = list(original.attn.qkv.weight.chunk(3)) + [original.attn.proj.weight,
              original.mlp.linear_fc1.weight,original.mlp.linear_fc2.weight]
    result = [b._rpu_q_w,b._rpu_k_w,b._rpu_v_w,b.attn.proj.weight,
              b.mlp.linear_fc1.weight,b.mlp.linear_fc2.weight]
    attrs = ['_rpu_q_ws','_rpu_k_ws','_rpu_v_ws','_rpu_o_ws','_rpu_fc1_ws','_rpu_fc2_ws']
    for i,(source,actual,attr) in enumerate(zip(raw,result,attrs)):
        q,s = quantize_linear_per_channel(source,bits=8)
        swizzle = tp_row_swizzle_mc_weight if i in (3,5) else tp_col_swizzle_mc_weight
        assert actual.dtype == torch.int8 and torch.equal(actual,swizzle(q,num_cores=8,dwidth=1))
        scale = getattr(b,attr)
        assert scale.dtype == torch.float16 and scale.shape == (source.shape[0],)
        assert torch.equal(scale,s) and torch.isfinite(scale).all()
    # Conversion never alters the original bias or already-installed owners.
    assert torch.equal(b._rpu_q_b,original.attn.qkv.bias[:1024])
    owners = [t.data_ptr() for t in result]
    vision._convert_vision_block_weights_for_rpu(b,16,1024,_g05_w8a16=True)
    assert owners == [t.data_ptr() for t in result]
    with pytest.raises(RuntimeError,match='immutable'):
        vision._convert_vision_block_weights_for_rpu(b,16,1024)
    fp16 = deepcopy(original)
    vision._convert_vision_block_weights_for_rpu(fp16,16,1024)
    assert fp16._rpu_q_w.dtype == torch.float16
    assert torch.equal(fp16._rpu_q_w,tp_col_swizzle_mc_weight(raw[0],num_cores=8,dwidth=2))
    assert not hasattr(fp16,'_rpu_q_ws')
    bad = deepcopy(original)
    with torch.no_grad(): bad.mlp.linear_fc2.weight[0,0] = float('nan')
    with pytest.raises(ValueError,match='finite pristine'):
        vision._convert_vision_block_weights_for_rpu(bad,16,1024,_g05_w8a16=True)
    assert not hasattr(bad,'_rpu_qwen3_5_vision_conversion_started')
    # Four lanes keep the same per-channel codes, swizzled for the reduced Vision owner.
    four = deepcopy(original)
    vision._convert_vision_block_weights_for_rpu(four,16,1024,num_cores=4,_g05_w8a16=True)
    for i,(source,actual) in enumerate(zip(raw,[four._rpu_q_w,four._rpu_k_w,four._rpu_v_w,
            four.attn.proj.weight,four.mlp.linear_fc1.weight,four.mlp.linear_fc2.weight])):
        q,_scale = quantize_linear_per_channel(source,bits=8)
        swizzle = tp_row_swizzle_mc_weight if i in (3,5) else tp_col_swizzle_mc_weight
        assert torch.equal(actual,swizzle(q,num_cores=4,dwidth=1))
    with pytest.raises(ValueError,match='num_cores must be 4 or 8'):
        vision._convert_vision_block_weights_for_rpu(deepcopy(original),16,1024,num_cores=6,_g05_w8a16=True)
