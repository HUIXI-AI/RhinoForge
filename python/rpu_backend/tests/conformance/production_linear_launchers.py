"""Real production-launcher fixtures; no model construction or checkpoint load.

Sparse weights give a separate CPU semantic reference without reproducing the
launcher register writer. Floating arithmetic differences are diagnostics;
layout/guard/replay and sparse linear identities are structural checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
import torch
import rpu_backend
from rpu_backend.runtime.weights import transform_linear_weight


def raw(x):
    return x.contiguous().view(torch.uint8).reshape(x.shape[0], -1)


def activation(cores, rows, k, arm):
    cc = torch.arange(cores)[:, None, None]
    mm = torch.arange(rows)[None, :, None]
    kk = torch.arange(k)[None, None, :]
    return (((kk * 7 + mm * 3 + cc * 5 + arm * 11) % 31 - 15).float() / 32).half()


def sparse_weight(n, k, cores, col, dtype, seed, resident=False):
    gn, gk = (n * cores, k) if col else (n, k * cores)
    weight = torch.zeros(gn, gk, dtype=dtype)
    indices = (torch.arange(gn) * 13 + seed) % k
    indices[-1] = k - 1
    signs = (1 - 2 * (torch.arange(gn) % 2)).to(dtype)
    amplitude = 0.25 if dtype == torch.float16 else 1
    if col:
        weight[torch.arange(gn), indices] = signs * amplitude
    else:
        for core in range(cores):
            weight[torch.arange(gn), indices + core*k] = signs * amplitude
    scale = ((torch.arange(gn) % 3 + 2).float() / 16).half()
    packed = transform_linear_weight(weight, int(col), cores).to('rpu')
    sr = torch.empty(0) if dtype == torch.float16 else scale.to('rpu')
    def expected(x):
        out=[]
        for core in range(cores):
            sl = slice(core*n, (core+1)*n) if col else slice(None)
            factors = signs[sl].float() * (amplitude if dtype == torch.float16 else scale[sl].float())
            out.append((x[core, :, indices[sl]].float() * factors).half())
        return torch.stack(out)
    compact = torch.stack([raw(transform_linear_weight(weight[c*n:(c+1)*n], 1, 1).reshape(1, -1))[0]
                           for c in range(cores)]) if col else None
    if resident and not col:
        compact = torch.stack([transform_linear_weight(weight[:,c*k:(c+1)*k].contiguous(), 1, 1)
                               .contiguous().view(torch.uint8).reshape(-1) for c in range(cores)])
    return packed, sr, expected, compact


def collective_fixture(name):
    cores=8
    profile=name
    assert name == 'qwen_o_norm'
    rows,n,k=1,2560,512
    weight,scale,oracle,_=sparse_weight(n,k,cores,False,torch.int8,3)
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm)
        partial=oracle(x)
        residual=activation(cores,rows,n,arm+5)
        gamma=((torch.arange(n)%5+12).float()/16).half().repeat(cores,1)
        # Collective consumes the corresponding owner shard of residual;
        # use one replicated residual to expose cross-core sum without an
        # ambiguous per-core logical model input.
        residual[:]=residual[0].clone()
        value=(partial.float().sum(0)+residual[0].float()).half()
        norm=(value.float()*torch.rsqrt(value.float().square().mean(-1,keepdim=True)+1e-6)*gamma[0].float()).half()
        zero=torch.zeros_like(partial)
        im=[raw(x),raw(zero),raw(residual),raw(zero),raw(gamma)];indices=[3,2]
        images.append(im);expected.append([value.repeat(cores,1,1),norm.repeat(cores,1,1)])
    return profile,rows,cores,0,[weight,scale],images,indices,expected,False


def a8_fixture():
    cores,rows,n,k=8,272,2048,2048
    w,sc,oracle,_=sparse_weight(n,k,cores,True,torch.int8,3)
    uw,usc,uoracle,_=sparse_weight(n,k,cores,True,torch.int8,41)
    owners=[w,uw,sc.cpu().float().to('rpu'),usc.cpu().float().to('rpu')]
    images=[];expected=[]
    for arm in range(2):
        cc=torch.arange(cores)[:,None,None];mm=torch.arange(rows)[None,:,None];kk=torch.arange(k)[None,None,:]
        codes=((kk*7+mm*3+cc*5+arm*11)%255-127).to(torch.int8)
        codes[:,:,0]=127
        x=(codes.half()/256).half()
        y,up=oracle(x),uoracle(x)
        images.append([raw(x),torch.zeros(cores,rows*2080+768,dtype=torch.uint8),raw(torch.zeros_like(y))])
        expected.append([(torch.nn.functional.gelu(y.float(),approximate='tanh').half()*up).half()])
    return 'pi_a8_online',rows,cores,0,owners,images,[2],expected,False


def a8_split_fixture():
    cores,rows,n,k=8,272,2048,2048
    print('PHASE a8_split gate weight',flush=True)
    w,sc,oracle,_=sparse_weight(n,k,cores,True,torch.int8,3)
    print('PHASE a8_split up weight',flush=True)
    uw,usc,uoracle,_=sparse_weight(n,k,cores,True,torch.int8,41)
    owners=[w,uw,sc.cpu().float().to('rpu'),usc.cpu().float().to('rpu')]
    images=[];expected=[]
    for arm in range(2):
        print('PHASE a8_split CPU arm '+str(arm),flush=True)
        cc=torch.arange(cores)[:,None,None];mm=torch.arange(2*rows)[None,:,None];kk=torch.arange(k)[None,None,:]
        codes=((kk*7+mm*3+cc*5+arm*11)%255-127).to(torch.int8)
        codes[:,:,0]=127
        sx=(2.0**((mm[:,:,0]+cc[:,:,0]+arm)%3-8)).half()
        x=(codes.float()*sx[:,:,None].float()).half()
        y,up=oracle(x),uoracle(x)
        combined=(torch.nn.functional.gelu(y.float(),approximate='tanh').half()*up).half()
        padded=torch.zeros(cores,2*rows,2080,dtype=torch.int8);padded[:,:,:k]=codes
        # Native reads 2*M half values plus a 30-byte vector tail. Keep the
        # CPU-mapped SPM copy/guard boundary 256-byte aligned as well: a libc
        # unaligned wide store to device memory can SIGBUS before emission.
        scale_bytes=((2*rows*2+30+255)//256)*256
        scale=torch.zeros(cores,scale_bytes//2,dtype=torch.float16);scale[:,:2*rows]=sx
        zero=torch.zeros(cores,rows,n,dtype=torch.float16)
        images.append([raw(padded),raw(scale),torch.zeros(cores,2*256*2048,dtype=torch.uint8),raw(zero),raw(zero)])
        expected.append([combined[:,:rows],combined[:,rows:]])
    return 'pi_a8_split',rows,cores,0,owners,images,[3,4],expected,False


def w4_geglu_fixture(group=32):
    from rpu_backend.quant.int4_pgrp_pack import swizzle_pack_int4_pgrp,swizzle_int4_pgrp_scale,w4_storage_shape
    cores,rows,n,k=8,50,4096,1024
    nn=torch.arange(n);weights=[];scales=[];projections=[]
    for seed in (3,41):
        indices=(nn*13+seed)%k;indices[-1]=k-1
        signs=(1-2*((nn+seed//3)%2)).to(torch.int8)
        q=torch.zeros(n,k,dtype=torch.int8);q[nn,indices]=signs
        logical_scale=(((torch.arange(k//group)[:,None]*3+nn[None,:]+seed)%5+1).half()/32)
        weights.append(swizzle_pack_int4_pgrp(q,1,cores).view(w4_storage_shape(n,k,1,cores)).to('rpu'))
        scales.append(swizzle_int4_pgrp_scale(logical_scale,group,1,cores).to('rpu'))
        projections.append((indices,signs,logical_scale[indices//group,nn]))
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);projected=[]
        for indices,signs,scale in projections:
            percore=[]
            for core in range(cores):
                sl=slice(core*(n//cores),(core+1)*(n//cores))
                percore.append((x[core,:,indices[sl]].float()*signs[sl].float()*scale[sl].float()).half())
            projected.append(torch.stack(percore))
        gate,up=projected
        y=(torch.nn.functional.gelu(gate.float(),approximate='tanh').half()*up).half()
        images.append([raw(x),raw(torch.zeros_like(y))]);expected.append([y])
    return 'pi_w4_geglu' if group==32 else 'pi_w4_geglu_n64',rows,cores,0,weights+scales,images,[1],expected,False


def grouped_fixture(name):
    from rpu_backend.quant.fp8 import pack_grouped_weight,pack_grouped_scale
    cores,experts,rows=8,32,204
    dtype=torch.float16 if name.endswith('fp16') else torch.int8
    down='ordered' in name or 'shuffle' in name
    expert_y4='expert_y4' in name
    fused='swiglu' in name
    n,k=(768,64) if down else (128,768)
    gn,gk=(n,k*cores) if down else (n*cores,k)
    w=torch.zeros(experts,gn,gk,dtype=dtype)
    nn=torch.arange(gn)
    indices=torch.stack([(nn*13+e*7+3)%k for e in range(experts)])
    signs=torch.stack([1-2*((nn+e)%2) for e in range(experts)])
    scales=torch.stack([((nn+e)%3+2).half()/16 for e in range(experts)])
    for e in range(experts):
        for c in range(cores if down else 1):
            w[e,nn,indices[e]+c*k]=(signs[e]*(0.25 if dtype==torch.float16 else 1)).to(dtype)
    owners=[pack_grouped_weight(w,partition='row' if down else 'col').to('rpu'),
            torch.empty(0) if dtype==torch.float16 else pack_grouped_scale(scales).to('rpu')]
    active=[0,7,16,31]
    counts=torch.zeros(128,dtype=torch.int16);counts[active]=51
    tokens=torch.zeros(256,dtype=torch.int16)
    inverse=torch.zeros(64,32,dtype=torch.int16)
    for a,e in enumerate(active):
        ids=(torch.arange(51)*7+e)%51
        tokens[a*51:(a+1)*51]=ids.short()
        inverse[ids,e]=torch.arange(a*51+1,(a+1)*51+1).short()
    route=((torch.arange(64)[:,None]*3+torch.arange(32)[None,:])%7+1).half()/16
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,204 if down or expert_y4 else 51,k,arm)
        y=torch.empty(cores,204,n,dtype=torch.float16)
        for a,e in enumerate(active):
            for c in range(cores):
                sl=slice(None) if down else slice(c*n,(c+1)*n)
                xx=x[c,a*51:(a+1)*51] if down or expert_y4 else x[c,tokens[a*51:(a+1)*51].long()]
                factors=signs[e,sl]*(0.25 if dtype==torch.float16 else scales[e,sl])
                y[c,a*51:(a+1)*51]=(xx[:,indices[e,sl]].float()*factors).half()
        ct=raw(counts.repeat(cores,1));zero=torch.zeros_like(y)
        if down:
            im=[raw(x),ct,raw(zero)];indices_out=[2];ex=[y]
            if 'ordered' in name:
                shared=activation(cores,51,768,arm+5)
                acc=torch.zeros_like(shared)
                for a,e in enumerate(active):
                    task=inverse[:51,e].long()-1
                    acc=(acc+y[:,task]).half()
                acc=(acc+shared).half()
                im.extend([raw(inverse.repeat(cores,1,1)),raw(shared),raw(torch.zeros_like(shared))]);indices_out=[2,5];ex=[y,acc]
        elif expert_y4:
            im=[raw(x),ct,raw(zero)];indices_out=[2];ex=[y]
        else:
            tok=raw(tokens.repeat(cores,1))
            if fused:
                gate,up=y[:,:,:64],y[:,:,64:]
                weighted=[]
                for a,e in enumerate(active): weighted.append(route[tokens[a*51:(a+1)*51].long(),e])
                factor=torch.cat(weighted)[None,:,None]
                out=(torch.nn.functional.silu(gate.float()).half()*up).half()
                out=(out*factor).half()
                im=[raw(x),tok,raw(route.repeat(cores,1,1)),ct,raw(torch.zeros_like(out))];indices_out=[4];ex=[out]
            else: im=[raw(x),tok,ct,raw(zero)];indices_out=[3];ex=[y]
        images.append(im);expected.append(ex)
    profile='lingbot_ordered' if 'ordered' in name else 'lingbot_down_shuffle' if down else 'lingbot_grouped_expert_y4' if expert_y4 else 'lingbot_grouped_swiglu' if fused else 'lingbot_grouped'
    return profile,51,cores,2 if dtype==torch.float16 else 1,owners,images,indices_out,expected,not fused


def qkv_fixture(name):
    cores=8;dtype=torch.float16 if name.endswith('fp16') else torch.int8
    owner='owner' in name
    if name.startswith('gr00t_qkv'): profile,rows,k,ns='gr00t_qkv',41,1536,[192]*3
    elif name in ('lingbot_prefill_qkv_w8','lingbot_prefill_qkv_fp16'): profile,rows,k,ns='lingbot_prefill_qkv',224,2560,[512,128,128]
    elif name.startswith('lingbot_qkv'): profile,rows,k,ns='lingbot_qkv',51,768,[512,128,128]
    elif name.startswith('g05_kv_owner'): profile,rows,k,ns='g05_kv_owner',32,1024,[256,256]
    else: profile,rows,k,ns='pi_kv_owner',50,1024,[256,256]
    weights=[];scales=[];oracles=[]
    for projection,n in enumerate(ns):
        w,sc,oracle,_=sparse_weight(n,k,cores,True,dtype,3+projection*19)
        weights.append(w);scales.append(sc);oracles.append(oracle)
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm)
        ex=[];biases=[]
        for projection,(n,oracle) in enumerate(zip(ns,oracles)):
            y=oracle(x)
            if profile=='g05_kv_owner':
                owner0=0 if projection==0 else 2  # K owners 0/4; V owners 2/6.
                y=torch.cat([y[owner0:owner0+1].repeat(4,1,1),y[owner0+4:owner0+5].repeat(4,1,1)])
            elif profile=='pi_kv_owner':
                y=torch.stack([y[c,:,torch.cat([torch.arange(c*16,c*16+16),torch.arange(c*16+128,c*16+144)])] for c in range(cores)])
            elif profile!='lingbot_prefill_qkv':
                bias=((torch.arange(cores*n).reshape(cores,n)*3+projection*5)%17-8).half()/128
                y=(y+bias[:,None,:]).half();biases.append(raw(bias))
            ex.append(y)
        images.append([raw(x)]+[raw(torch.zeros_like(y)) for y in ex]+biases);expected.append(ex)
    return profile,rows,cores,int(name.endswith('acc32')),weights+scales,images,list(range(1,len(ns)+1)),expected,True


def qwen_planar_fixture():
    cores,rows,n,k=8,192,512,2048
    weight,scale,oracle,_=sparse_weight(n,k,cores,True,torch.int8,3)
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);dense=oracle(x)
        planar=torch.cat([dense[:,:,:256].reshape(cores,-1),
                          dense[:,:,256:384].reshape(cores,-1),
                          dense[:,:,384:].reshape(cores,-1)],dim=1)
        images.append([raw(x),raw(torch.zeros_like(planar))]);expected.append([planar])
    return 'qwen_qkv_planar',rows,cores,0,[weight,scale],images,[1],expected,True


def prefill_kv_owner_fixture(dtype=torch.float16, rows=272, signed_scales=False):
    cores,k=8,2048
    weights=[];scales=[];projections=[]
    for seed in (3,41):
        nn=torch.arange(256);indices=(nn*13+seed)%k;indices[-1]=k-1
        signs=(1-2*((nn+seed//3)%2)).to(dtype)
        logical=torch.zeros(256,k,dtype=dtype)
        logical[nn,indices]=signs/4 if dtype==torch.float16 else signs
        weights.append(transform_linear_weight(logical.repeat(cores,1),1,cores).to('rpu'))
        scale=((nn*3+seed)%5+1).half()/32
        if signed_scales:
            assert dtype == torch.int8
            scale *= (1-2*((nn//3+seed)%2)).half()
        scales.append(torch.empty(0) if dtype==torch.float16 else scale.repeat(cores).to('rpu'))
        factors=signs.float()*(.25 if dtype==torch.float16 else scale.float())
        projections.append((indices,factors))
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);outputs=[]
        for indices,factors in projections:
            parts=[]
            for core in range(cores):
                stripe=torch.cat([torch.arange(core*16,core*16+16),
                                  torch.arange(core*16+128,core*16+144)])
                parts.append((x[core,:,indices[stripe]].float()*factors[stripe]).half())
            outputs.append(torch.stack(parts))
        images.append([raw(x)]+[raw(torch.zeros_like(y)) for y in outputs]);expected.append(outputs)
    return 'pi_prefill_kv_owner',rows,cores,0,weights+scales,images,[1,2],expected,True


def resident_preload_fixture(name):
    cores=8
    rows,n,k,col,packed_bytes=(50,512,1024,True,524288) if name=='pi_resident_preload' else (32,1024,512,False,344064)
    weight,scale,oracle,_=sparse_weight(n,k,cores,col,torch.int8,3)
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);y=oracle(x)
        images.append([raw(x),torch.zeros(cores,packed_bytes,dtype=torch.uint8),raw(torch.zeros_like(y))])
        expected.append([y])
    return name,rows,cores,0,[weight,scale],images,[2],expected,True


def q_rope_fixture():
    cores,rows,n,k,position=8,50,256,1024,13
    weight,scale,oracle,_=sparse_weight(n,k,cores,True,torch.int8,3)
    phase=(torch.arange(80)[:,None]*3+torch.arange(128)[None,:])%4
    cosine=torch.tensor([1,0,-1,0],dtype=torch.float16)[phase]
    sine=torch.tensor([0,1,0,-1],dtype=torch.float16)[phase]
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);y=oracle(x)
        co=cosine[position:position+rows];si=sine[position:position+rows]
        rotated=torch.cat([y[:,:,:128]*co-y[:,:,128:]*si,
                           y[:,:,128:]*co+y[:,:,:128]*si],dim=-1).half()
        images.append([raw(x),raw(torch.zeros_like(y))]);expected.append([rotated])
    return 'pi_q_rope',rows,cores,position,[weight,scale,cosine.to('rpu'),sine.to('rpu')],images,[1],expected,True


def packed_gateup_fixture(dtype):
    cores,rows,n,k=8,51,96,768
    gate,gs,go,_=sparse_weight(n,k,cores,True,dtype,3)
    up,us,uo,_=sparse_weight(n,k,cores,True,dtype,41)
    # Weights are complete Gate then Up swizzles, while per-channel scales
    # are [core, Gate96, Up96]. Neither is a normal [core, N192] weight.
    weight=torch.cat([gate.cpu().flatten(),up.cpu().flatten()]).reshape(1536,k).to('rpu')
    scale=torch.empty(0) if dtype==torch.float16 else torch.cat([gs.cpu().reshape(cores,n),us.cpu().reshape(cores,n)],dim=1).flatten().to('rpu')
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);g,u=go(x),uo(x)
        y=(torch.nn.functional.silu(g.float()).half()*u).half()
        images.append([raw(x),raw(torch.zeros_like(y))]);expected.append([y])
    return 'lingbot_packed_gateup',rows,cores,0,[weight,scale],images,[1],expected,False


def wall_owner_qkv_fixture(dtype):
    cores,rows,k,position=8,20,1024,13
    weights=[];scales=[];oracles=[];biases=[]
    for p,n in enumerate((256,128,128)):
        w,sc,oracle,_=sparse_weight(n,k,cores,True,dtype,3+p*38)
        weights.append(w);scales.append(sc);oracles.append(oracle)
        biases.append(((torch.arange(cores*n).reshape(cores,n)*3+p*5)%17-8).half()/128)
    phase=(torch.arange(80)[:,None]*3+torch.arange(64)[None,:])%4
    co=torch.tensor([1,0,-1,0],dtype=torch.float16)[phase]
    si=torch.tensor([0,1,0,-1],dtype=torch.float16)[phase]
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);outputs=[]
        for p,oracle in enumerate(oracles):
            y=(oracle(x)+biases[p][:,None,:]).half()
            if p<2:
                z=y.reshape(cores,rows,-1,128)
                c=co[position:position+rows,None,:];s=si[position:position+rows,None,:]
                y=torch.cat([z[:,:,:,:64]*c-z[:,:,:,64:]*s,
                             z[:,:,:,64:]*c+z[:,:,:,:64]*s],dim=-1).reshape_as(y)
            if p:
                owner=0 if p==1 else 2
                y=torch.cat([y[owner:owner+1].repeat(4,1,1),y[owner+4:owner+5].repeat(4,1,1)])
            outputs.append(y)
        images.append([raw(x)]+[raw(torch.zeros_like(y)) for y in outputs]+[raw(b) for b in biases])
        expected.append(outputs)
    return 'wall_action_qkv_owner',rows,cores,position,weights+scales+[co.to('rpu'),si.to('rpu')],images,[1,2,3],expected,True


def vision_bias_fixture(rows,dtype,triplet):
    cores,n,k=8,448,1280
    slabs=3 if triplet else 1
    gw,gs,go,_=sparse_weight(n,k,cores,True,dtype,3)
    uw,us,uo,_=sparse_weight(n,k,cores,True,dtype,41)
    gb=((torch.arange(cores*n).reshape(cores,n)*3)%17-8).half()/128
    ub=((torch.arange(cores*n).reshape(cores,n)*5+7)%19-9).half()/256
    images=[];expected=[]
    for arm in range(2):
        xs=[activation(cores,rows,k,arm+3*p) for p in range(slabs)];ys=[]
        for x in xs:
            # Up channel zero on core zero reads K=41. Force that projection
            # to zero while its independent Up bias remains -2/256: a missing
            # or aliased Up-bias slot then fails the sign/zero semantic check.
            x[0,:,41]=0
            g=(go(x)+gb[:,None,:]).half();u=(uo(x)+ub[:,None,:]).half()
            ys.append((torch.nn.functional.silu(g.float()).half()*u).half())
        images.append([raw(x) for x in xs]+[raw(torch.zeros_like(y)) for y in ys]+[raw(gb),raw(ub)])
        expected.append(ys)
    profile='wall_vision_triplet' if triplet else 'wall_vision_shared_bias'
    return profile,rows,cores,0,[gw,uw,gs,us],images,list(range(slabs,2*slabs)),expected,False


def bf16_output_fixture():
    cores,rows,n,k=8,288,1280,448
    w,_,oracle,_=sparse_weight(n,k,cores,False,torch.float16,3)
    bias=((torch.arange(cores*n).reshape(cores,n)*3)%17-8).half()/128
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);y=(oracle(x).float()+bias[:,None,:].float()).bfloat16()
        images.append([raw(x),raw(torch.zeros_like(y)),raw(bias)]);expected.append([y])
    return 'wall_bf16_output',rows,cores,0,[w],images,[1],expected,True


def wall_ring_fixture(local):
    cores,rows,n,k=8,32 if local else 20,1024,256
    dtype=torch.int8 if local else torch.float16
    w,sc,oracle,_=sparse_weight(n,k,cores,False,dtype,3)
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);partial=oracle(x)
        residual=activation(cores,rows,n,arm+5);residual[:]=residual[0].clone()
        gamma=((torch.arange(n)%5+12).float()/16).half().repeat(cores,1)
        value=(partial.float().sum(0)+residual[0].float()).half()
        norm=(value.float()*torch.rsqrt(value.float().square().mean(-1,keepdim=True)+1e-6)*gamma[0].float()).half()
        compact=torch.zeros(cores,(rows+7)//8,n,dtype=torch.float16)
        local_residual=torch.zeros_like(compact)
        for c in range(cores):
            count=rows//8+int(c<rows%8);start=c*(rows//8)+min(c,rows%8)
            compact[c,:count]=value[start:start+count]
            local_residual[c,:count]=residual[0,start:start+count]
        zero=torch.zeros_like(partial)
        im=[raw(x),raw(zero)]
        if local:im += [raw(local_residual),raw(zero),raw(gamma)];indices=[2,3]
        else:im += [raw(residual),raw(torch.zeros_like(compact)),raw(zero),raw(gamma)];indices=[3,4]
        images.append(im);expected.append([compact,norm.repeat(cores,1,1)])
    return 'wall_action_ring',rows,cores,int(local),[w,sc],images,indices,expected,False


def gate_kv_owner_fixture():
    cores,rows,n,k=8,32,256,1024
    weights=[];scales=[];oracles=[]
    for p in range(3):
        w,sc,oracle,_=sparse_weight(n,k,cores,True,torch.int8,3+p*38)
        weights.append(w);scales.append(sc);oracles.append(oracle)
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);ys=[]
        for p,oracle in enumerate(oracles):
            y=oracle(x)
            if p:
                owners=(0,4) if p==1 else (2,6)
                for core in range(cores):
                    if core not in owners:y[core].zero_()
            ys.append(y)
        images.append([raw(x)]+[raw(torch.zeros_like(y)) for y in ys]);expected.append(ys)
    return 'g05_gate_kv_owner',rows,cores,0,weights+scales,images,[1,2,3],expected,True


def compact_kv_fixture():
    cores,rows,n,k=8,50,32,1024
    w,sc,oracle,_=sparse_weight(n,k,cores,True,torch.int8,3)
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);y=oracle(x)
        images.append([raw(x),raw(torch.zeros_like(y))]);expected.append([y])
    return 'pi_compact_kv',rows,cores,0,[w,sc],images,[1],expected,True


def paired_geglu_fixture():
    cores,rows,n,k=8,272,2048,2048
    gw,gs,go,_=sparse_weight(n,k,cores,True,torch.int8,3)
    uw,us,uo,_=sparse_weight(n,k,cores,True,torch.int8,41)
    images=[];expected=[]
    for arm in range(2):
        xs=[activation(cores,rows,k,arm+3*p) for p in range(2)]
        ys=[(torch.nn.functional.gelu(go(x).float(),approximate='tanh').half()*uo(x)).half() for x in xs]
        images.append([raw(x) for x in xs]+[raw(torch.zeros_like(y)) for y in ys]);expected.append(ys)
    return 'pi_paired_geglu',rows,cores,0,[gw,uw,gs,us],images,[2,3],expected,False


def fixture(name):
    if name in PAIRED_LINEAR_REMAINING_CASES: return paired_linear_remaining_fixture(name)
    if name in ('shared_followup_p171','shared_followup_p172','shared_followup_p173'): return shared384_native_fixture(name)
    if name in PAIRED_LINEAR_ROW_CASES: return paired_linear_row_fixture(name)
    if name in QKV_REMAINING_CASES: return qkv_remaining_fixture(name)
    if name in KV_OWNER_W8_LARGE_CASES: return kv_owner_w8_large_fixture(name)
    if name in KV_OWNER_TAIL_CASES: return kv_owner_tail_fixture(name)
    if name in SHARED_GELU_FOLLOWUP_CASES: return shared_gelu_followup_fixture()
    if name in PAIRED_GELU_W8_FOLLOWUP_CASES: return paired_geglu_w8_followup_fixture(name)
    if name in PAIRED_GELU_FOLLOWUP_CASES: return paired_geglu_followup_fixture(name)
    if name in QKV_PAIR_FOLLOWUP_CASES: return qkv_pair_followup_fixture(name)
    if name in ROW_CACHE_FOLLOWUP_CASES: return row_cache_followup_fixture(*ROW_CACHE_FOLLOWUP_PROFILES[int(name.rsplit('p',1)[1])])
    if name in SHARED_FOLLOWUP_CASES: return shared_followup_fixture(name)
    if name in GROUPED_DEVICE_CASES+GROUPED_FOLLOWUP_CASES: return grouped_device_fixture(name)
    if name in HEADER_CASES: return header_fixture(name)
    if name in GAP_CASES: return gap_fixture(name)
    if name=='g05_gate_kv_owner_w8': return gate_kv_owner_fixture()
    if name=='pi_compact_kv_w8': return compact_kv_fixture()
    if name=='pi_paired_geglu_w8': return paired_geglu_fixture()
    if name in ('wall_ring_m20_fp16','wall_ring_m32_local_w8'): return wall_ring_fixture(name.endswith('w8'))
    if name.startswith('wall_action_qkv_owner_'): return wall_owner_qkv_fixture(torch.float16 if name.endswith('fp16') else torch.int8)
    if name.startswith('wall_vision_triplet_'):
        return vision_bias_fixture(576,torch.float16 if name.endswith('fp16') else torch.int8,True)
    if name.startswith('wall_vision_shared_w8_m'):
        return vision_bias_fixture(int(name.rsplit('m',1)[1]),torch.int8,False)
    if name=='wall_bf16_output': return bf16_output_fixture()
    if name in ('pi_resident_preload','g05_partial_preload'): return resident_preload_fixture(name)
    if name=='pi_q_rope': return q_rope_fixture()
    if name.startswith('lingbot_packed_gateup_'): return packed_gateup_fixture(torch.float16 if name.endswith('fp16') else torch.int8)
    if name=='qwen_qkv_planar': return qwen_planar_fixture()
    if name=='pi_prefill_kv_owner_m272_fp16': return prefill_kv_owner_fixture()
    if name=='pi_prefill_kv_owner_m272_w8': return prefill_kv_owner_fixture(torch.int8)
    if name in KV_OWNER_FOLLOWUP_CASES:
        return prefill_kv_owner_fixture(rows=int(name.split('_m')[1].split('_')[0]))
    if name in KV_OWNER_REMAINING_CASES:
        rows,dtype=KV_OWNER_REMAINING_PROFILES[int(name.rsplit('p',1)[1])]
        return prefill_kv_owner_fixture(torch.float16 if dtype=='fp16' else torch.int8,
                                       rows=rows,signed_scales=dtype=='w8')
    if name.startswith(('gr00t_qkv','lingbot_qkv','lingbot_prefill_qkv','g05_kv_owner','pi_kv_owner')): return qkv_fixture(name)
    if name == 'qwen_o_norm': return collective_fixture(name)
    if name=='pi_a8_online': return a8_fixture()
    if name=='pi_a8_split': return a8_split_fixture()
    if name=='pi_w4_geglu_g32': return w4_geglu_fixture()
    if name.startswith(('lingbot_grouped','lingbot_ordered','lingbot_down_shuffle')): return grouped_fixture(name)
    cores, mode, col, activation_kind = 8, 0, True, 'linear'
    dtype=torch.int8
    if name.startswith('g05_qgate'):
        profile, rows, n, k = 'g05_qgate_w8',32,256,1024
        mode=int(name.endswith('acc32')); pair=True
    elif name == 'g05_gateup_fp16':
        profile,rows,n,k='g05_gateup_fp16',32,512,1024;dtype=torch.float16;pair=True
    elif name == 'g05_vision_fc2_w8':
        profile,rows,n,k='g05_vision_fc2',768,1024,512;col=False;pair=False
    elif name.startswith('hyvla_'):
        rows,n,k=51,256,1024
        dtype=torch.float16 if 'fp16' in name else torch.int8
        resident='resident' in name
        profile=('hyvla_resident_fp16' if resident else 'hyvla_gateup_fp16') if dtype==torch.float16 else 'hyvla_gateup_w8'
        mode=(2 if 'fused' in name else 1) if resident and dtype==torch.int8 else 0
        pair=resident and 'fused' not in name
        activation_kind='linear' if pair else 'swiglu'
    elif name.startswith('wall_action'):
        profile,rows,n,k='wall_action_gateup',32 if 'm32' in name else 20,256,1024
        dtype=torch.float16 if 'fp16' in name else torch.int8
        mode=int('resident' in name);pair=True
    elif name.startswith('wall_prefill'):
        profile,rows,n,k='wall_prefill_gateup',640 if 'm640' in name else 320,1376,2048
        pair=False;activation_kind='swiglu'
    elif name=='qwen_gateup':
        profile,rows,n,k='qwen_gateup',1,1216,2560;pair=False;activation_kind='swiglu'
    elif name.startswith('pi_pair'):
        profile,rows,n,k='pi_pair',400 if 'col' in name else 272,256 if 'col' in name else 2048,2048 if 'col' in name else 256
        col='col' in name;mode=int(col);dtype=torch.float16 if 'fp16' in name else torch.int8;pair=False
    elif name.startswith('pi_down_pair'):
        profile,rows,n,k='pi_down_pair',272,2048,2048;col=False;dtype=torch.float16 if 'fp16' in name else torch.int8;pair=False
    elif name.startswith('lingbot_routed'):
        profile,rows,n,k='lingbot_routed',51,2048,768;pair=False;dtype=torch.float16 if 'fp16' in name else torch.int8
    elif name=='lingbot_router':
        profile,rows,n,k='lingbot_router',51,32,768;cores=1;dtype=torch.float16;pair=False
    elif name.startswith('lingbot_fc1'):
        profile,rows,n,k='lingbot_fc1',768,512,1024;pair=False;dtype=torch.float16 if 'fp16' in name else torch.int8;activation_kind='gelu'
    elif name.startswith('pi_vision_fc'):
        profile,rows='pi_vision_fc',768;col='fc1' in name;mode=int(col);n,k=(544,1152) if col else (1152,544)
        dtype=torch.float16 if 'fp16' in name else torch.int8;pair=False;activation_kind='linear'
    else: raise ValueError(name)
    w,scale,oracle,compact=sparse_weight(n,k,cores,col,dtype,3)
    owners=[w,scale]
    if pair or activation_kind=='swiglu':
        uw,uscale,uoracle,ucompact=sparse_weight(n,k,cores,col,dtype,41)
        owners=[w,uw] if profile=='g05_gateup_fp16' else [w,uw,scale,uscale]
        if profile=='hyvla_gateup_fp16': owners=[w,uw]
        if profile=='hyvla_resident_fp16': owners=[]
    images=[];expected=[]
    output_indices=[1,2] if pair else [1]
    for arm in range(2):
        x=activation(cores,rows,k,arm)
        y=oracle(x)
        zero=torch.zeros(cores,rows,n,dtype=torch.float16)
        im=[raw(x),raw(zero)]
        ex=[y]
        if pair:
            im.append(raw(zero));ex.append(uoracle(x))
        if activation_kind=='swiglu':
            up=uoracle(x)
            ex=[(torch.nn.functional.silu(y.float()).half()*up).half()]
            if profile=='wall_prefill_gateup': im.append(torch.zeros(cores,573440,dtype=torch.uint8))
        if profile in ('hyvla_resident_fp16',) or ('resident' in name and profile in ('hyvla_gateup_w8','wall_action_gateup')):
            if len(im)==2: im.append(raw(zero))
            im.extend([compact,ucompact])
        if profile in ('pi_pair','pi_down_pair'):
            x1=activation(cores,rows,k,arm+2)
            im=[raw(x),raw(x1),raw(zero),raw(zero)];output_indices=[2,3];ex=[y,oracle(x1)]
        if profile=='lingbot_routed':
            counts=torch.tensor([0,1,17,51]*8,dtype=torch.int16).repeat(cores,1)
            im=[raw(x),raw(counts),raw(zero)];output_indices=[2]
            ex=[y.reshape(cores,rows,32,64).masked_fill(counts[:,None,:,None]==0,0).reshape_as(y)]
        if profile=='lingbot_router':
            z=torch.zeros(cores,64,n,dtype=torch.float16)
            im=[raw(x),raw(z),raw(z)];output_indices=[1,2]
            logits=z.clone();logits[:,:rows]=y
            scores=torch.ones_like(z);scores[:,:rows]=torch.sigmoid(y.float()).half();ex=[logits,scores]
        if profile in ('lingbot_fc1','pi_vision_fc','g05_vision_fc2'):
            bias=((torch.arange(cores*n).reshape(cores,n)%7-3).float()/128).half()
            if not col: bias[1:]=0
            im.append(raw(bias));y=(y+bias[:,None,:]).half()
            ex=[torch.nn.functional.gelu(y.float(),approximate='tanh').half() if activation_kind=='gelu' else y]
        images.append(im);expected.append(ex)
    return profile,rows,cores,mode,owners,images,output_indices,expected,activation_kind=='linear'


CASES=['g05_qgate_acc16','g05_qgate_acc32','g05_gateup_fp16','g05_vision_fc2_w8',
       'hyvla_gateup_fp16','hyvla_gateup_w8','hyvla_resident_fp16','hyvla_resident_w8','hyvla_resident_fused_w8',
       'wall_action_w8','wall_action_m32_fp16','wall_action_resident_w8','wall_prefill_m320_w8',
       'qwen_gateup','pi_pair_row_w8','pi_pair_row_fp16','pi_pair_col_w8','pi_down_pair_w8',
       'lingbot_router','lingbot_routed_w8','lingbot_routed_fp16','lingbot_fc1_w8',
       'pi_vision_fc1_w8','pi_vision_fc2_w8','qwen_o_norm','pi_a8_online',
       'lingbot_grouped_w8','lingbot_grouped_fp16','lingbot_grouped_expert_y4_w8','lingbot_grouped_swiglu_w8','lingbot_grouped_swiglu_fp16',
       'lingbot_ordered_w8','lingbot_ordered_fp16','lingbot_down_shuffle_w8',
       'gr00t_qkv_w8','lingbot_qkv_w8','lingbot_qkv_fp16','lingbot_prefill_qkv_w8','lingbot_prefill_qkv_fp16','g05_kv_owner_w8_acc16','g05_kv_owner_w8_acc32',
       'g05_kv_owner_fp16','pi_kv_owner_w8','pi_kv_owner_fp16','pi_w4_geglu_g32','pi_a8_split',
       'qwen_qkv_planar','pi_prefill_kv_owner_m272_fp16','pi_prefill_kv_owner_m272_w8',
       'pi_resident_preload','g05_partial_preload','pi_q_rope','lingbot_packed_gateup_w8','lingbot_packed_gateup_fp16',
       'wall_action_qkv_owner_fp16','wall_action_qkv_owner_w8','wall_vision_triplet_w8','wall_vision_triplet_fp16',
       'wall_vision_shared_w8_m324','wall_vision_shared_w8_m576','wall_vision_shared_w8_m672','wall_bf16_output',
       'wall_ring_m20_fp16','wall_ring_m32_local_w8',
       'g05_gate_kv_owner_w8','pi_compact_kv_w8','pi_paired_geglu_w8']


# Each entry below reaches a previously unexecuted production launcher body.
# Keep these separate from the frozen final66 selection and its source snapshot.
GAP_CASES=['gr00t_ff1_resident','g05_gateup_resident_fp16','g05_gateup_resident_w8','g05_o_resident_w8',
           'g05_qgate_fp16','g05_gateup_fp16_acc16','g05_gateup_w8_acc32',
           'g05_text_gateup_fp16_acc16','g05_text_gateup_fp16_acc32','g05_text_gateup_w8_acc32',
           'g05_vision_qkv_fp16','g05_vision_qkv_w8','lingbot_vision_qkv_w8',
           'wall_qkv_w8','wall_vision_qkv_w8','generic_spm_fp16_acc32','wall_vision_row_w8',
           'qwen_prefill_swiglu_w8','wall_base_prefill_swiglu_fp16','lingbot_prefill_swiglu_w8',
           'pi_prefill_geglu_w8','pi_denoise_geglu_w8','pi_w4_geglu_n64_g128','pi_w4_geglu_m50_g128',
           'wall_down_c320x2','wall_oproj_c320x2']

HEADER_CASES=['hyvla_header_vlm_gateup','hyvla_header_kv_fp16','hyvla_header_kv_w8',
              'hyvla_header_o_resident','hyvla_header_ring_fp16','hyvla_header_ring_w8',
              'generic_grouped_uint4_acc16','generic_grouped_uint4_acc32']

GROUPED_DEVICE_CASES=['generic_grouped_device_m96','generic_grouped_device_m97',
                      'generic_grouped_device_m128','generic_grouped_w8_device_m48',
                      'generic_grouped_w8_device_m49','generic_grouped_w8_device_m256']

# Explicitly separate FP8 storage/launcher mode from signed INT8 evidence.
GROUPED_FOLLOWUP_PROFILES={
    'fp16_acc32_n80':dict(n=256,k=512,mode=2,acc32=True,cold=80,original=128),
    'fp16_acc32_n32':dict(n=1024,k=256,mode=2,acc32=True,cold=80,original=256),
    'w8_acc16_n48':dict(n=256,k=512,mode=1,acc32=False,cold=48,original=128),
    'w8_acc32_n48':dict(n=256,k=512,mode=1,acc32=True,cold=96,original=128),
    'fp8_acc32_n112':dict(n=1024,k=256,mode=5,acc32=True,cold=48,original=256),
    'fp8_acc16_n48':dict(n=256,k=512,mode=5,acc32=False,cold=48,original=128),
    'fp8_acc32_n48':dict(n=256,k=512,mode=5,acc32=True,cold=96,original=128),
}
GROUPED_FOLLOWUP_CASES=[f'generic_grouped_followup_{name}_m{rows}'
    for name,spec in GROUPED_FOLLOWUP_PROFILES.items()
    for rows in (spec['cold'],spec['cold']+1,spec['original'])]

KV_OWNER_FOLLOWUP_CASES=['pi_prefill_kv_owner_m304_fp16','pi_prefill_kv_owner_m400_fp16']

# A separate batch: exact GP9 rows, with signed K/V scales in the W8 cases.
KV_OWNER_REMAINING_PROFILES={249:(432,'fp16'),250:(448,'fp16'),253:(272,'w8'),255:(304,'w8')}
KV_OWNER_REMAINING_CASES=[f'pi_prefill_kv_owner_remaining_p{profile}'
    for profile in KV_OWNER_REMAINING_PROFILES]

QKV_PAIR_FOLLOWUP_CASES=['wall_qkv_pair_p118','wall_qkv_pair_p136']
# Rows, full output N, global weight K; p004 additionally selects no-bias Causal O.
ROW_CACHE_FOLLOWUP_PROFILES={
    4:(512,1024,2048,True),
    39:(1728,1280,1280),40:(512,1280,1280),41:(768,1280,1280),
    42:(324,1280,1280),43:(672,1280,1280),
    44:(1728,1280,3584),45:(512,1280,3584),46:(768,1280,3584),
    47:(324,1280,3584),48:(672,1280,3584),
}
ROW_CACHE_FOLLOWUP_CASES=[f'wall_row_cache_p{profile:03d}'
    for profile in ROW_CACHE_FOLLOWUP_PROFILES]


def qkv_pair_followup_fixture(name):
    assert name in QKV_PAIR_FOLLOWUP_CASES
    owner=name.endswith('p118');cores,rows,k=8,640,2048
    weights=[];scales=[];oracles=[]
    # K seed652 places a nonzero K2047 term on its last active even owner.
    for n,seed in zip((256,128,128),(3,652,41)):
        w,sc,oracle,_=sparse_weight(n,k,cores,True,torch.float16,seed)
        weights.append(w);scales.append(sc);oracles.append(oracle)
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);initial=[];outputs=[];biases=[]
        for projection,(n,oracle) in enumerate(zip((256,128,128),oracles)):
            bias=((torch.arange(cores*n).reshape(cores,n)*3+projection*5+arm)%17-8).half()/128
            full=(oracle(x)+bias[:,None,:]).half();biases.append(raw(bias))
            if owner and projection:
                before=activation(cores,rows,64,arm+7+projection)
                y=before.clone()
                for core in range(cores):
                    if core%2==projection-1:
                        start=64*((core%4)//2)
                        y[core]=full[core,:,start:start+64]
            else:
                y=full;before=torch.zeros_like(y)
            initial.append(raw(before));outputs.append(y)
        images.append([raw(x)]+initial+biases);expected.append(outputs)
    return 'wall_qkv_pair_fp16',rows,cores,int(owner),weights+scales,images,[1,2,3],expected,True


def row_cache_followup_fixture(rows,n,global_k,causal=False):
    cores=8;assert global_k%cores==0
    if causal:assert (rows,n,global_k)==(512,1024,2048)
    k=global_k//cores
    weight,scale,oracle,_=sparse_weight(n,k,cores,False,torch.int8,3)
    scale_sign=(1-2*((torch.arange(n)//3)%2)).half()
    scale=(scale.cpu()*scale_sign).to('rpu')
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm)
        bias=((torch.arange(cores*n).reshape(cores,n)*3+arm)%17-8).half()/128
        bias[1:]=0
        if causal:bias.zero_()
        y=(oracle(x)*scale_sign[None,None,:]+bias[:,None,:]).half()
        images.append([raw(x),raw(torch.zeros_like(y))]+([] if causal else [raw(bias)]));expected.append([y])
    profile='causal_o_row_followup_w8' if causal else 'wall_vision_row_followup_w8'
    return profile,rows,cores,0,[weight,scale],images,[1],expected,True


# Exact public caller coverage; production selector decisions are frozen later.
SHARED_FOLLOWUP_PROFILES={
    15:(176,1216,2560,'w8','qwen_prefill_swiglu_w8','fused'),
    20:(256,1216,2560,'w8','qwen_prefill_swiglu_w8','fused'),
    160:(320,1376,2048,'fp16','wall_prefill_gateup','fused'),
    161:(640,1376,2048,'fp16','wall_prefill_gateup','fused'),
    166:(320,1376,2048,'w8','wall_prefill_gateup','fused'),
    167:(640,1376,2048,'w8','wall_prefill_gateup','fused'),
    168:(336,1376,2048,'w8','wall_base_shared_raw','both_raw'),
    169:(336,1376,2048,'w8','wall_base_shared_silu','fused_and_up'),
    170:(336,1376,2048,'w8','wall_base_shared_no_up','fused'),
    171:(384,1376,2048,'w8','wall_base_shared_raw','both_raw'),
    172:(384,1376,2048,'w8','wall_base_shared_silu','fused_and_up'),
    173:(384,1376,2048,'w8','wall_base_shared_no_up','fused'),
    175:(224,1376,2048,'w8','wall_base_shared_no_up','fused'),
    177:(256,1376,2048,'w8','wall_base_shared_no_up','fused'),
    178:(320,1376,2048,'fp16','wall_base_prefill_swiglu_fp16','fused'),
    179:(320,1376,2048,'w8','wall_base_shared_no_up','fused'),
}
SHARED_FOLLOWUP_CASES=[f'shared_followup_p{p:03d}' for p in SHARED_FOLLOWUP_PROFILES]


PAIRED_GELU_FOLLOWUP_PROFILES={180:272,183:288,186:304,189:320,192:336,195:400,198:416,201:432,204:448,207:464}
PAIRED_GELU_FOLLOWUP_CASES=[f'paired_geglu_followup_p{p}' for p in PAIRED_GELU_FOLLOWUP_PROFILES]


PAIRED_GELU_W8_FOLLOWUP_PROFILES={210:272,213:288,216:304,219:320,222:336,225:400,228:416,231:432,234:448,237:464}
PAIRED_GELU_W8_FOLLOWUP_CASES=[f'paired_geglu_w8_followup_p{p}' for p in PAIRED_GELU_W8_FOLLOWUP_PROFILES]


KV_OWNER_TAIL_PROFILES={243:(288,'fp16'),248:(416,'fp16'),251:(464,'fp16'),
                        254:(288,'w8'),256:(320,'w8'),257:(336,'w8')}
KV_OWNER_TAIL_CASES=[f'pi_prefill_kv_owner_tail_p{p}' for p in KV_OWNER_TAIL_PROFILES]
SHARED_GELU_FOLLOWUP_CASES=['shared_gelu_followup_p155']

KV_OWNER_W8_LARGE_PROFILES={258:(400,'w8'),259:(416,'w8'),260:(432,'w8'),261:(448,'w8'),262:(464,'w8')}
KV_OWNER_W8_LARGE_CASES=[f'pi_prefill_kv_owner_large_p{p}' for p in KV_OWNER_W8_LARGE_PROFILES]


def kv_owner_w8_large_fixture(name):
    rows,precision=KV_OWNER_W8_LARGE_PROFILES[int(name.rsplit('p',1)[1])]
    dtype=torch.float16 if precision=='fp16' else torch.int8
    cores,k=8,2048;weights=[];scales=[];projections=[]
    for seed in (3,41):
        nn=torch.arange(256);indices=(nn*13+seed)%k
        # Both D16 stripes on each physical core contain a nonzero K2047 term.
        indices[torch.arange(16)*16+15]=k-1
        signs=(1-2*((nn+seed//3)%2)).to(dtype)
        logical=torch.zeros(256,k,dtype=dtype)
        logical[nn,indices]=signs/4 if dtype==torch.float16 else signs
        weights.append(transform_linear_weight(logical.repeat(cores,1),1,cores).to('rpu'))
        scale=((nn*3+seed)%5+1).half()/32
        if dtype==torch.int8:scale*=(1-2*((nn//3+seed)%2)).half()
        scales.append(torch.empty(0) if dtype==torch.float16 else scale.repeat(cores).to('rpu'))
        factors=signs.float()*(.25 if dtype==torch.float16 else scale.float())
        projections.append((indices,factors))
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);outputs=[]
        for indices,factors in projections:
            parts=[]
            for core in range(cores):
                stripe=torch.cat([torch.arange(core*16,core*16+16),torch.arange(core*16+128,core*16+144)])
                parts.append((x[core,:,indices[stripe]].float()*factors[stripe]).half())
            outputs.append(torch.stack(parts))
        images.append([raw(x)]+[raw(torch.full_like(y,float('nan'))) for y in outputs]);expected.append(outputs)
    return 'pi_prefill_kv_owner',rows,cores,0,weights+scales,images,[1,2],expected,True


def kv_owner_tail_fixture(name):
    rows,precision=KV_OWNER_TAIL_PROFILES[int(name.rsplit('p',1)[1])]
    dtype=torch.float16 if precision=='fp16' else torch.int8
    cores,k=8,2048;weights=[];scales=[];projections=[]
    for seed in (3,41):
        nn=torch.arange(256);indices=(nn*13+seed)%k
        # Both D16 stripes on each physical core contain a nonzero K2047 term.
        indices[torch.arange(16)*16+15]=k-1
        signs=(1-2*((nn+seed//3)%2)).to(dtype)
        logical=torch.zeros(256,k,dtype=dtype)
        logical[nn,indices]=signs/4 if dtype==torch.float16 else signs
        weights.append(transform_linear_weight(logical.repeat(cores,1),1,cores).to('rpu'))
        scale=((nn*3+seed)%5+1).half()/32
        if dtype==torch.int8:scale*=(1-2*((nn//3+seed)%2)).half()
        scales.append(torch.empty(0) if dtype==torch.float16 else scale.repeat(cores).to('rpu'))
        factors=signs.float()*(.25 if dtype==torch.float16 else scale.float())
        projections.append((indices,factors))
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);outputs=[]
        for indices,factors in projections:
            parts=[]
            for core in range(cores):
                stripe=torch.cat([torch.arange(core*16,core*16+16),torch.arange(core*16+128,core*16+144)])
                parts.append((x[core,:,indices[stripe]].float()*factors[stripe]).half())
            outputs.append(torch.stack(parts))
        images.append([raw(x)]+[raw(torch.full_like(y,float('nan'))) for y in outputs]);expected.append(outputs)
    return 'pi_prefill_kv_owner',rows,cores,0,weights+scales,images,[1,2],expected,True


def shared_gelu_followup_fixture():
    cores,rows,n,k=8,400,2048,2048
    gw,gs,gate=paired_geglu_w8_followup_weight(n,k,cores,3,0)
    uw,us,up=paired_geglu_w8_followup_weight(n,k,cores,41,1)
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm)
        y=(torch.nn.functional.gelu(gate(x).float(),approximate='tanh').half()*up(x)).half()
        images.append([raw(x),raw(torch.full_like(y,float('nan')))]);expected.append([y])
    return 'pi_prefill_geglu_w8',rows,cores,0,[gw,uw,gs,us],images,[1],expected,False


def paired_geglu_w8_followup_weight(n,k,cores,seed,projection):
    channels=torch.arange(n*cores)
    indices=(channels*13+seed)%k
    indices[(torch.arange(cores)+1)*n-1]=k-1
    signs=(1-2*(channels%2)).to(torch.int8)
    logical=torch.zeros(n*cores,k,dtype=torch.int8)
    logical[channels,indices]=signs
    # Independent signed per-channel Gate/Up scale owners exercise scale roles.
    scale=(((channels%3+2).float()/16)*(1-2*((channels//3+projection)%2))/(2**projection)).half()
    packed=transform_linear_weight(logical,1,cores).to('rpu')
    owner_scale=scale.to('rpu')
    factors=(signs.half()*scale).half()
    def expected(x):
        return torch.stack([(x[core,:,indices[core*n:(core+1)*n]].float()*
            factors[core*n:(core+1)*n].float()).half() for core in range(cores)])
    return packed,owner_scale,expected


def paired_geglu_w8_followup_fixture(name):
    rows=PAIRED_GELU_W8_FOLLOWUP_PROFILES[int(name.rsplit('p',1)[1])]
    cores,n,k=8,2048,2048
    gate_weight,gate_scale,gate=paired_geglu_w8_followup_weight(n,k,cores,3,0)
    up_weight,up_scale,up=paired_geglu_w8_followup_weight(n,k,cores,41,1)
    images=[];expected=[]
    for arm in range(2):
        xs=[activation(cores,rows,k,arm+3*pair) for pair in range(2)]
        ys=[(torch.nn.functional.gelu(gate(x).float(),approximate='tanh').half()*up(x)).half() for x in xs]
        images.append([raw(x) for x in xs]+[raw(torch.full_like(y,float('nan'))) for y in ys])
        expected.append(ys)
    return 'pi_paired_geglu',rows,cores,0,[gate_weight,up_weight,gate_scale,up_scale],images,[2,3],expected,False


def paired_geglu_followup_weight(n,k,cores,seed):
    channels=torch.arange(n*cores)
    indices=(channels*13+seed)%k
    # Both projections explicitly exercise K2047 on every physical core.
    indices[(torch.arange(cores)+1)*n-1]=k-1
    factors=((1-2*(channels%2)).float()/4).half()
    logical=torch.zeros(n*cores,k,dtype=torch.float16)
    logical[channels,indices]=factors
    packed=transform_linear_weight(logical,1,cores).to('rpu')
    def expected(x):
        return torch.stack([(x[core,:,indices[core*n:(core+1)*n]].float()*
            factors[core*n:(core+1)*n].float()).half() for core in range(cores)])
    return packed,expected


def paired_geglu_followup_fixture(name):
    rows=PAIRED_GELU_FOLLOWUP_PROFILES[int(name.rsplit('p',1)[1])]
    cores,n,k=8,2048,2048
    gate_weight,gate=paired_geglu_followup_weight(n,k,cores,3)
    up_weight,up=paired_geglu_followup_weight(n,k,cores,41)
    images=[];expected=[]
    for arm in range(2):
        xs=[activation(cores,rows,k,arm+3*pair) for pair in range(2)]
        ys=[(torch.nn.functional.gelu(gate(x).float(),approximate='tanh').half()*up(x)).half() for x in xs]
        # Poison both output images to expose unwritten final rows/channels.
        images.append([raw(x) for x in xs]+[raw(torch.full_like(y,float('nan'))) for y in ys])
        expected.append(ys)
    # Empty owners become undefined scales in the unchanged public C++ branch.
    return 'pi_paired_geglu',rows,cores,0,[gate_weight,up_weight,torch.empty(0),torch.empty(0)],images,[2,3],expected,False


def shared_followup_weight(n,k,cores,dtype,seed):
    # The last channel on every column-parallel core probes the final K term.
    # Keep the common sparse_weight fixture unchanged for all earlier batches.
    channels=torch.arange(n*cores)
    indices=(channels*13+seed)%k
    indices[(torch.arange(cores)+1)*n-1]=k-1
    signs=(1-2*(channels%2)).to(dtype)
    amplitude=0.25 if dtype==torch.float16 else 1
    logical=torch.zeros(n*cores,k,dtype=dtype)
    logical[channels,indices]=signs*amplitude
    packed=transform_linear_weight(logical,1,cores).to('rpu')
    scale=((channels%3+2).float()/16).half()
    owner_scale=torch.empty(0) if dtype==torch.float16 else scale.to('rpu')
    def expected(x):
        pieces=[]
        for core in range(cores):
            sl=slice(core*n,(core+1)*n)
            factors=signs[sl].float()*(amplitude if dtype==torch.float16 else scale[sl].float())
            pieces.append((x[core,:,indices[sl]].float()*factors).half())
        return torch.stack(pieces)
    return packed,owner_scale,expected


def shared_followup_fixture(name):
    rows,n,k,precision,profile,output_mode=SHARED_FOLLOWUP_PROFILES[int(name.rsplit('p',1)[1])]
    cores=8;dtype=torch.float16 if precision=='fp16' else torch.int8
    gw,gs,gate_oracle=shared_followup_weight(n,k,cores,dtype,3)
    uw,us,up_oracle=shared_followup_weight(n,k,cores,dtype,41)
    # Distinct nonuniform scale owners expose swapped Gate/Up scale bindings.
    if precision=='w8':us=(us.cpu()/2).to('rpu')
    images=[];expected=[];indices=[1,2] if output_mode!='fused' else [1]
    for arm in range(2):
        x=activation(cores,rows,k,arm)
        gate=gate_oracle(x);up=up_oracle(x)
        if precision=='w8':up=(up/2).half()
        fused=(torch.nn.functional.silu(gate.float()).half()*up).half()
        outputs=[gate,up] if output_mode=='both_raw' else [fused,up] if output_mode=='fused_and_up' else [fused]
        im=[raw(x)]+[raw(torch.zeros_like(y)) for y in outputs]
        if profile=='wall_prefill_gateup':im.append(torch.zeros(cores,573440,dtype=torch.uint8))
        if profile=='wall_base_shared_no_up':
            # Public no-Up API still requires a disjoint full Up owner. The
            # conformance wrapper checks non-output SPM images are immutable.
            im.append(raw(activation(cores,rows,n,arm+9)))
        images.append(im);expected.append(outputs)
    return profile,rows,cores,0,[gw,uw,gs,us],images,indices,expected,output_mode=='both_raw'


def grouped_device_fixture(name):
    from rpu_backend.quant.fp8 import pack_grouped_weight,pack_grouped_scale
    rows=int(name.rsplit('m',1)[1])
    if name in GROUPED_FOLLOWUP_CASES:
        variant=name.removeprefix('generic_grouped_followup_').rsplit('_m',1)[0]
        spec=GROUPED_FOLLOWUP_PROFILES[variant]
        profile='generic_grouped_followup_'+variant
    else:
        w8='_w8_' in name
        spec=dict(n=1024 if w8 else 256,k=256 if w8 else 512,
                  mode=1 if w8 else 2,cold=48 if w8 else 96,original=256 if w8 else 128)
        profile='generic_grouped_w8_device_m' if w8 else 'generic_grouped_device_m'
    w8=spec['mode']!=2;fp8=spec['mode']==5
    cores,n,k,experts=8,spec['n'],spec['k'],3
    col=rows!=spec['original']
    gn,gk=(cores*n,k) if col else (n,cores*k)
    nn=torch.arange(gn)
    weights=torch.zeros(experts,gn,gk,dtype=torch.uint8 if fp8 else torch.int8 if w8 else torch.float16)
    indices=torch.stack([(nn*13+e*7+3)%k for e in range(experts)])
    indices[:,-1]=k-1
    signs=torch.stack([1-2*((nn+e)%2) for e in range(experts)])
    scales=torch.stack([((nn+e)%3+2).half()/16 for e in range(experts)])
    encoded=torch.where(signs>0,0x38,0xb8).to(torch.uint8) if fp8 else None
    # These exact E4M3 codes decode to +/-1, whereas INT8 would see 56/-72.
    decoded=encoded.contiguous().view(torch.float8_e4m3fn).float() if fp8 else signs.float()
    for e in range(experts):
        for c in range(1 if col else cores):
            weights[e,nn,indices[e]+c*k]=(encoded[e] if fp8 else signs[e].to(weights.dtype)
                if w8 else (signs[e].float()/4).half())
    owners=[pack_grouped_weight(weights,partition='col' if col else 'row').to('rpu')]
    if w8:owners.append(pack_grouped_scale(scales).to('rpu'))
    images=[];expected=[]
    for arm in range(2):
        # Same retained graph sees a hot expert, then balanced counts. A device
        # choice frozen during BUILD cannot satisfy this B/A/B replay probe.
        values=([0,0,rows] if rows==spec['cold']+1 else [0,1,rows-1]) if arm==0 else [rows//3,rows//3,rows-2*(rows//3)]
        counts=torch.zeros(128,dtype=torch.int16);counts[:experts]=torch.tensor(values)
        assert int(counts.sum())==rows
        x=activation(cores,rows,k,arm)
        bias=(((torch.arange(cores*n).reshape(cores,n)*3+arm)%17-8).float()/128).half()
        y=torch.empty(cores,rows,n,dtype=torch.float16)
        offset=0
        for e,count in enumerate(values):
            for c in range(cores):
                sl=slice(c*n,(c+1)*n) if col else slice(None)
                factor=decoded[e,sl]*(scales[e,sl].float() if w8 else .25)
                projected=(x[c,offset:offset+count,indices[e,sl]].float()*factor).half()
                y[c,offset:offset+count]=(projected+bias[c]).half()
            offset+=count
        images.append([raw(x),raw(counts.repeat(cores,1)),raw(torch.zeros_like(y)),raw(bias)])
        expected.append([y])
    return profile,rows,cores,int(not col),owners,images,[2],expected,True


def header_fixture(name):
    if name.startswith('generic_grouped_uint4'):
        from rpu_backend.quant.grouped_uint4 import pack_grouped_uint4,pack_grouped_uint4_metadata
        cores,rows,n,k,experts,group=8,51,48,256,5,32
        nn=torch.arange(cores*n);qq=torch.arange(k)
        codes=torch.stack([((nn[:,None]*7+qq[None,:]*3+e*5)%7+5).byte() for e in range(experts)])
        scales=torch.stack([((torch.arange(k//group)[:,None]+nn[None,:]+e)%4+1).half()/64 for e in range(experts)])
        scales[:,:,::17]=0
        owners=[pack_grouped_uint4(codes,partition='col',num_cores=cores).to('rpu')]
        owners += [pack_grouped_uint4_metadata(v,partition='col',num_cores=cores).to('rpu')
                   for v in (scales,torch.full_like(scales,-8))]
        counts=torch.zeros(128,dtype=torch.int16);counts[:5]=torch.tensor([1,0,17,33,0])
        row_expert=torch.repeat_interleave(torch.arange(experts),counts[:experts].long()).tolist()
        images=[];expected=[]
        for arm in range(2):
            x=torch.zeros(cores,rows,k,dtype=torch.float16);y=torch.empty(cores,rows,n,dtype=torch.float16)
            for c in range(cores):
                sl=slice(c*n,(c+1)*n)
                for row,e in enumerate(row_expert):
                    kk=(row*31+c*17+arm*7)%k;value=(0.5 if (row+arm)%2 else -0.25)
                    x[c,row,kk]=value
                    y[c,row]=((codes[e,sl,kk].float()-8)*scales[e,kk//group,sl].float()*value).half()
            images.append([raw(x),raw(counts.repeat(cores,1)),raw(torch.zeros_like(y))]);expected.append([y])
        return 'generic_grouped_uint4',rows,cores,int(name.endswith('acc32')),owners,images,[2],expected,True
    cores=8;dtype=torch.float16 if name.endswith('fp16') else torch.int8
    if 'ring' in name:
        rows,n,k=51,1024,256
        weight,scale,oracle,_=sparse_weight(n,k,cores,False,dtype,3)
        images=[];expected=[]
        for arm in range(2):
            x=activation(cores,rows,k,arm);partial=oracle(x)
            residual=activation(cores,64,n,arm+5);residual[:]=residual[0].clone()
            gamma=((torch.arange(n)%5+12).float()/16).half().repeat(cores,1)
            # Ring produces only the 51 logical rows. Its raw owner has 64
            # rows for the subsequent eight-row-per-core RMSNorm partition.
            raw_y=torch.zeros(64,n,dtype=torch.float16)
            raw_y[:rows]=(partial.float().sum(0)+residual[0,:rows].float()).half()
            norm=(raw_y.float()*torch.rsqrt(raw_y.float().square().mean(-1,keepdim=True)+1e-5)*gamma[0].float()).half()
            images.append([raw(x),raw(torch.zeros_like(partial)),raw(residual),
                           raw(torch.zeros_like(residual)),raw(gamma)])
            expected.append([raw_y.repeat(cores,1,1),norm.repeat(cores,1,1)])
        return name,rows,cores,0,[weight,scale],images,[3,2],expected,False
    rows,n,k,col=(240,768,2048,True) if name=='hyvla_header_vlm_gateup' else (
        (51,1024,256,False) if name=='hyvla_header_o_resident' else (51,128,1024,True))
    resident=name=='hyvla_header_o_resident'
    weight,scale,oracle,compact=sparse_weight(n,k,cores,col,dtype,3,resident=resident)
    owners=[scale] if resident else [weight,scale]
    if not resident:
        uw,usc,uoracle,_=sparse_weight(n,k,cores,col,dtype,41)
        owners=[weight,uw,scale,usc]
    images=[];expected=[];indices=[1]
    for arm in range(2):
        x=activation(cores,rows,k,arm);y=oracle(x);im=[raw(x),raw(torch.zeros_like(y))];ex=[y]
        if resident:im.append(compact)
        elif name=='hyvla_header_vlm_gateup':ex=[(torch.nn.functional.silu(y.float()).half()*uoracle(x)).half()]
        else:im.append(raw(torch.zeros_like(y)));ex.append(uoracle(x));indices=[1,2]
        images.append(im);expected.append(ex)
    profile='hyvla_header_kv' if 'header_kv_' in name else name
    return profile,rows,cores,0,owners,images,indices,expected,name!='hyvla_header_vlm_gateup'


def gap_fixture(name):
    if name=='pi_w4_geglu_n64_g128': return w4_geglu_fixture(128)
    if name=='pi_w4_geglu_m50_g128':
        _,*fixture=w4_geglu_fixture(128)
        return ('pi_w4_geglu',*fixture)
    cores=8
    dtype=torch.float16 if 'fp16' in name else torch.int8
    qkv_shapes={
        'g05_vision_qkv_fp16':(256,1024,[128]*3),
        'g05_vision_qkv_w8':(256,1024,[128]*3),
        'lingbot_vision_qkv_w8':(768,1024,[128]*3),
        'wall_qkv_w8':(320,2048,[256,128,128]),
        'wall_vision_qkv_w8':(1728,1280,[160]*3)}
    if name in qkv_shapes:
        rows,k,ns=qkv_shapes[name];weights=[];scales=[];oracles=[]
        for p,n in enumerate(ns):
            w,sc,oracle,_=sparse_weight(n,k,cores,True,dtype,3+p*19)
            weights.append(w);scales.append(sc);oracles.append(oracle)
        images=[];expected=[]
        for arm in range(2):
            x=activation(cores,rows,k,arm);ex=[];biases=[]
            for p,(n,oracle) in enumerate(zip(ns,oracles)):
                bias=((torch.arange(cores*n).reshape(cores,n)*3+p*5+arm)%17-8).half()/128
                ex.append((oracle(x)+bias[:,None,:]).half());biases.append(raw(bias))
            images.append([raw(x)]+[raw(torch.zeros_like(y)) for y in ex]+biases);expected.append(ex)
        return name,rows,cores,0,weights+scales,images,[1,2,3],expected,True
    # Local N and K are independently specified by each real admitted profile.
    # The two projections use different sparse locations, signs and scales.
    specs={
        'gr00t_ff1_resident':(41,768,1536,True,'resident_bias'),
        'g05_gateup_resident_fp16':(32,512,1024,True,'resident'),
        'g05_gateup_resident_w8':(32,512,1024,True,'resident'),
        'g05_o_resident_w8':(32,1024,256,False,'resident'),
        'g05_qgate_fp16':(32,256,1024,True,'pair'),
        'g05_gateup_fp16_acc16':(32,512,1024,True,'pair'),
        'g05_gateup_w8_acc32':(32,512,1024,True,'pair'),
        'g05_text_gateup_fp16_acc16':(256,768,2048,True,'pair'),
        'g05_text_gateup_fp16_acc32':(256,768,2048,True,'pair'),
        'g05_text_gateup_w8_acc32':(256,768,2048,True,'pair'),
        'generic_spm_fp16_acc32':(17,48,256,True,'bias'),
        'wall_vision_row_w8':(324,1280,160,False,'bias'),
        'qwen_prefill_swiglu_w8':(16,1216,2560,True,'swiglu'),
        'wall_base_prefill_swiglu_fp16':(224,1376,2048,True,'swiglu'),
        'lingbot_prefill_swiglu_w8':(224,1216,2560,True,'swiglu'),
        'pi_prefill_geglu_w8':(400,2048,2048,True,'geglu'),
        'pi_denoise_geglu_w8':(50,512,1024,True,'geglu'),
        'wall_down_c320x2':(640,2048,1376,False,'split'),
        'wall_oproj_c320x2':(640,2048,256,False,'split')}
    rows,n,k,col,kind=specs[name]
    resident=kind.startswith('resident')
    w,sc,oracle,compact=sparse_weight(n,k,cores,col,dtype,3,resident=resident)
    owners=[w,sc]
    if kind in ('pair','swiglu','geglu'):
        uw,usc,uoracle,_=sparse_weight(n,k,cores,col,dtype,41)
        owners=[w,uw,sc,usc]
    if resident:
        owners=[] if dtype==torch.float16 else [sc]
        assert compact.shape==(cores,n*k*(2 if dtype==torch.float16 else 1))
    images=[];expected=[];indices=[1]
    for arm in range(2):
        x=activation(cores,rows,k,arm);y=oracle(x)
        im=[raw(x),raw(torch.zeros_like(y))];ex=[y]
        if kind=='pair':im.append(raw(torch.zeros_like(y)));ex.append(uoracle(x));indices=[1,2]
        if kind in ('swiglu','geglu'):
            nonlinear=(torch.nn.functional.silu(y.float()) if kind=='swiglu' else
                       torch.nn.functional.gelu(y.float(),approximate='tanh')).half()
            ex=[(nonlinear*uoracle(x)).half()]
        if resident:im.append(compact)
        if 'bias' in kind:
            bias=((torch.arange(cores*n).reshape(cores,n)*3+arm)%17-8).half()/128
            if not col:bias[1:]=0
            ex=[(y+bias[:,None,:]).half()];im.append(raw(bias))
        if kind=='split':
            im=[raw(x),raw(torch.zeros_like(y[:,:320])),raw(torch.zeros_like(y[:,320:]))]
            ex=[y[:,:320],y[:,320:]];indices=[1,2]
        images.append(im);expected.append(ex)
    return name,rows,cores,0,owners,images,indices,expected,kind not in ('swiglu','geglu')


def diagnostics(actual,expected):
    a,e=actual.double().flatten(),expected.double().flatten()
    error=(a-e).abs()
    maximum=error.max().item()
    # Descriptive distribution only: these bins never gate numerical PASS.
    histogram=(dict(edges=[maximum*i/8 for i in range(9)],
                    counts=[int(x) for x in torch.histc(error,bins=8,min=0,max=maximum).tolist()])
               if maximum else dict(edges=[0,0],counts=[error.numel()]))
    return dict(max_abs=maximum, mean_abs=error.mean().item(), mse=((a-e)**2).mean().item(),
                cosine=torch.nn.functional.cosine_similarity(a,e,dim=0).item(),
                element_count=error.numel(),exact_element_count=int(torch.count_nonzero(error==0)),
                absolute_error_histogram=histogram,
                nonzero=int(torch.count_nonzero(a)), expected_nonzero=int(torch.count_nonzero(e)))


def eager_w4_cases(out=None):
    from rpu_backend.quant.int4_pgrp_pack import (swizzle_pack_int4_pgrp,
        swizzle_int4_pgrp_scale,w4_storage_shape,pack_int4_per_channel_as_pgrp)
    reports=[]
    for local_n,group_size,per_channel in [(16,32,False),(48,32,False),(48,32,True)]:
        n,k,rows=local_n*8,256,17
        q=torch.zeros(n,k,dtype=torch.int8)
        nn=torch.arange(n);kk=torch.tensor([0,15,16,31,32,63,64,127,128,255])[nn%10]
        q[nn,kk]=((nn%7)-3).to(torch.int8)
        scale_n=((nn%5)+1).half()/64
        logical_scale=(((torch.arange(k//group_size)[:,None]*3+nn[None,:])%5+1).half()/64)
        if per_channel:
            packed,sc=pack_int4_per_channel_as_pgrp(q,scale_n,1,8,group_size)
            scale_at=scale_n
        else:
            packed=swizzle_pack_int4_pgrp(q,1,8)
            sc=swizzle_int4_pgrp_scale(logical_scale,group_size,1,8)
            scale_at=logical_scale[kk//group_size,nn]
        weight=packed.view(w4_storage_shape(n,k,1,8)).to('rpu');scale=sc.to('rpu')
        bias=((nn%7-3).half()/512).to('rpu');values=[];metrics=[];references=[]
        for arm in [0,1,0,1]:
            x=activation(1,rows,k,arm)
            expected=(x[:,:,kk].float()*(q[nn,kk].float()*scale_at.float())+bias.cpu().float()).half()
            actual=torch.ops.rpu.linear_w4a16(x.to('rpu'),weight,scale,bias,n).cpu()
            if out is not None:
                if len(references)<2:references.append(expected)
                torch.save(dict(values=[v.contiguous().view(torch.uint8) for v in values+[actual]],
                    expected=references,metadata=dict(case='eager_w4',local_n=local_n,group_size=group_size,
                        per_channel=per_channel,sequence=[0,1,0,1][:len(values)+1])),
                    out/f'eager_w4_n{local_n}_g{group_size}_pc{int(per_channel)}.tensors.pt')
            assert actual.shape==(1,rows,n)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            values.append(actual);metrics.append(diagnostics(actual,expected))
        assert torch.equal(values[0],values[2]) and torch.equal(values[1],values[3])
        assert not torch.equal(values[0],values[1])
        try: torch.ops.rpu.linear_w4a16(x.to('rpu'),weight,scale,bias,n-1)
        except RuntimeError as error: rejected=str(error)
        else: raise AssertionError('invalid logical N accepted')
        reports.append(dict(case='eager_w4',local_n=local_n,group_size=group_size,per_channel=per_channel,
                            physical_weight_shape=list(weight.shape),logical_output_shape=list(values[0].shape),
                            exact=True,invalid_logical_n_rejected=rejected,metrics=metrics))
    # The public eager entry deliberately admits only G32. Generic grouped
    # G64/G128 coverage is separate; reject these before DMA/launch here.
    for unsupported_group in (64,128):
        scales=torch.ones(k//unsupported_group,n,dtype=torch.float16)/64
        invalid=swizzle_int4_pgrp_scale(scales,unsupported_group,1,8).to('rpu')
        try: torch.ops.rpu.linear_w4a16(x.to('rpu'),weight,invalid,bias,n)
        except RuntimeError as error:
            assert 'G32' in str(error)
            reports.append(dict(case='eager_w4_group_rejected',group_size=unsupported_group,error=str(error)))
        else: raise AssertionError('unsupported eager W4 group accepted')
    return reports


def negative_owner_cases(out=None):
    profile,rows,cores,mode,owners,images,indices,expected,_=fixture('g05_qgate_acc16')
    reports=[]
    for label,index,replacement in [('cpu_weight',0,owners[0].cpu()),('wrong_scale_dtype',2,owners[2].cpu().to(torch.int8).to('rpu'))]:
        bad=list(owners);bad[index]=replacement
        try: torch.ops.rpu.test_production_linear_launchers(profile,bad,*images,indices,rows,cores,mode)
        except RuntimeError as error: reports.append(dict(case=label,rejected=str(error)))
        else: raise AssertionError('invalid production owner accepted: '+label)
    # A valid launch after both aborted BUILDs proves local cleanup remains usable.
    values,nodes,kargs,metadata=torch.ops.rpu.test_production_linear_launchers(profile,owners,*images,indices,rows,cores,mode)
    if out is not None:
        name='negative_owner_recovery_g05_qgate_acc16'
        torch.save(dict(values=values,expected=expected,metadata=json.loads(metadata)),out/(name+'.tensors.pt'))
        (out/(name+'.nodes.jsonl')).write_text(nodes)
        (out/(name+'.args.jsonl')).write_text(kargs)
    return reports


QKV_REMAINING_PROFILES={
    130:('fp16',1728,1280,(160,160,160),True),
    131:('fp16',224,2048,(256,128,128),False),
    132:('fp16',256,2048,(256,128,128),False),
    133:('fp16',320,2048,(256,128,128),False),
    134:('fp16',336,2048,(256,128,128),False),
    135:('fp16',384,2048,(256,128,128),False),
    137:('w8',1728,1280,(160,160,160),True),
    138:('w8',224,2048,(256,128,128),False),
    139:('w8',256,2048,(256,128,128),False),
    143:('w8',640,2048,(256,128,128),False),
}
QKV_REMAINING_CASES=[f'wall_qkv_remaining_p{pid:03d}' for pid in QKV_REMAINING_PROFILES]


def qkv_remaining_weight(n,k,cores,dtype,seed):
    assert n%16==0 and cores==8
    g=torch.arange(cores*n);indices=(g*13+seed)%k
    # Every D16 end probes final K, including every projection/core and N tail.
    indices[15::16]=k-1
    signs=(1-2*((g+seed)%2)).to(dtype)
    coefficients=signs*(0.25 if dtype==torch.float16 else (g%3+1).to(dtype))
    logical=torch.zeros(cores*n,k,dtype=dtype);logical[g,indices]=coefficients
    scale_values=torch.tensor([1,-2,3,-4,5,-6],dtype=torch.float16)/32
    scales=scale_values[(g//2+seed)%6]
    factors=coefficients.half() if dtype==torch.float16 else (coefficients.half()*scales).half()
    packed=transform_linear_weight(logical,1,cores).to('rpu')
    scale=torch.empty(0) if dtype==torch.float16 else scales.to('rpu')
    def project(x):
        return torch.stack([(x[c,:,indices[c*n:(c+1)*n]].float()*factors[c*n:(c+1)*n].float()).half() for c in range(cores)])
    return packed,scale,project


def qkv_remaining_fixture(name):
    assert name in QKV_REMAINING_CASES
    pid=int(name.rsplit('p',1)[1]);kind,rows,k,ns,vision=QKV_REMAINING_PROFILES[pid]
    cores=8;dtype=torch.float16 if kind=='fp16' else torch.int8
    weights=[];scales=[];projections=[]
    for n,seed in zip(ns,(3,41,97)):
        weight,scale,project=qkv_remaining_weight(n,k,cores,dtype,seed)
        weights.append(weight);scales.append(scale);projections.append(project)
    images=[];expected=[]
    for arm in range(2):
        x=activation(cores,rows,k,arm);outputs=[];biases=[]
        for projection,(n,project) in enumerate(zip(ns,projections)):
            bias=((torch.arange(cores*n).reshape(cores,n)*3+projection*5+arm)%17-8).half()/128
            outputs.append((project(x)+bias[:,None,:]).half());biases.append(raw(bias))
        images.append([raw(x)]+[raw(torch.full_like(y,float('nan'))) for y in outputs]+biases)
        expected.append(outputs)
    profile=('wall_vision_qkv_fp16_followup' if kind=='fp16' else 'wall_vision_qkv_w8') if vision else ('wall_qkv_pair_fp16' if kind=='fp16' else 'wall_qkv_w8')
    return profile,rows,cores,0,weights+scales,images,[1,2,3],expected,True



PAIRED_LINEAR_ROW_PROFILES={184:(288,'fp16'),187:(304,'fp16'),190:(320,'fp16'),193:(336,'fp16'),196:(400,'fp16'),199:(416,'fp16'),202:(432,'fp16'),205:(448,'fp16'),208:(464,'fp16'),214:(288,'int8'),217:(304,'int8'),223:(336,'int8'),226:(400,'int8'),229:(416,'int8'),232:(432,'int8'),235:(448,'int8'),238:(464,'int8')}
PAIRED_LINEAR_ROW_CASES=[f'paired_linear_row_followup_p{p}' for p in PAIRED_LINEAR_ROW_PROFILES]


def paired_linear_row_weight(n,k,cores,dtype):
    nn=torch.arange(n);cc=torch.arange(cores)[:,None]
    indices=(nn[None,:]*13+cc*17+3)%k
    indices[:,15::16]=k-1
    signs=1-2*((nn[None,:]+cc)%2)
    magnitude=(nn[None,:]+2*cc)%2+1
    coefficients=(signs*magnitude).to(dtype)
    if dtype==torch.float16:coefficients=coefficients/16
    logical=torch.zeros(n,k*cores,dtype=dtype)
    for core in range(cores):logical[nn,indices[core]+core*k]=coefficients[core]
    scale=torch.tensor([-3,-2,-1,1,2,3],dtype=torch.float16)[nn%6]/16
    factors=coefficients.half() if dtype==torch.float16 else (coefficients.half()*scale[None,:]).half()
    packed=transform_linear_weight(logical,0,cores).to('rpu')
    scale_owner=torch.empty(0) if dtype==torch.float16 else scale.to('rpu')
    def project(x):
        return torch.stack([(x[core,:,indices[core]].float()*factors[core].float()).half() for core in range(cores)])
    return packed,scale_owner,project


def paired_linear_row_fixture(name):
    rows,kind=PAIRED_LINEAR_ROW_PROFILES[int(name.rsplit('p',1)[1])]
    dtype=torch.float16 if kind=='fp16' else torch.int8
    cores,n,k=8,2048,256
    weight,scale,project=paired_linear_row_weight(n,k,cores,dtype)
    images=[];expected=[]
    for arm in range(2):
        x0=activation(cores,rows,k,arm);x1=activation(cores,rows,k,arm+2)
        poison=torch.full((cores,rows,n),float('nan'),dtype=torch.float16)
        images.append([raw(x0),raw(x1),raw(poison.clone()),raw(poison.clone())])
        expected.append([project(x0),project(x1)])
    return 'pi_pair',rows,cores,0,[weight,scale],images,[2,3],expected,True


PAIRED_LINEAR_REMAINING_PROFILES={120:(320,'fp16',1376,'wall_down_fp16_c320x2'),182:(272,'fp16',2048,'pi_down_pair'),185:(288,'fp16',2048,'pi_down_pair'),188:(304,'fp16',2048,'pi_down_pair'),191:(320,'fp16',2048,'pi_down_pair'),194:(336,'fp16',2048,'pi_down_pair'),197:(400,'fp16',2048,'pi_down_pair'),200:(416,'fp16',2048,'pi_down_pair'),203:(432,'fp16',2048,'pi_down_pair'),206:(448,'fp16',2048,'pi_down_pair'),209:(464,'fp16',2048,'pi_down_pair'),215:(288,'int8',2048,'pi_down_pair'),218:(304,'int8',2048,'pi_down_pair'),221:(320,'int8',2048,'pi_down_pair'),224:(336,'int8',2048,'pi_down_pair'),227:(400,'int8',2048,'pi_down_pair'),230:(416,'int8',2048,'pi_down_pair'),233:(432,'int8',2048,'pi_down_pair'),236:(448,'int8',2048,'pi_down_pair'),239:(464,'int8',2048,'pi_down_pair')}
PAIRED_LINEAR_REMAINING_CASES=[f'paired_linear_remaining_p{p}' for p in PAIRED_LINEAR_REMAINING_PROFILES]


def paired_linear_remaining_fixture(name):
    rows,kind,k,profile=PAIRED_LINEAR_REMAINING_PROFILES[int(name.rsplit('p',1)[1])]
    dtype=torch.float16 if kind=='fp16' else torch.int8
    cores,n=8,2048
    weight,scale,project=paired_linear_row_weight(n,k,cores,dtype)
    images=[];expected=[]
    for arm in range(2):
        x0=activation(cores,rows,k,arm);x1=activation(cores,rows,k,arm+2)
        poison=torch.full((cores,rows,n),float('nan'),dtype=torch.float16)
        inputs=[raw(torch.cat((x0,x1),dim=1))] if profile=='wall_down_fp16_c320x2' else [raw(x0),raw(x1)]
        images.append(inputs+[raw(poison.clone()),raw(poison.clone())])
        expected.append([project(x0),project(x1)])
    return profile,rows*2 if profile=='wall_down_fp16_c320x2' else rows,cores,0,[weight,scale],images,([1,2] if profile=='wall_down_fp16_c320x2' else [2,3]),expected,True


def shared384_native_fixture(name):
    data=shared_followup_fixture(name)
    profile,rows,cores,mode,owners,images,indices,expected,exact=data
    assert rows==384 and int(name.rsplit('p',1)[1]) in (171,172,173)
    for im in images:
        for index in indices:im[index]=raw(torch.full_like(im[index].view(torch.float16),float('nan')))
    return data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--conformance',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--case',action='append',choices=PAIRED_LINEAR_REMAINING_CASES+PAIRED_LINEAR_ROW_CASES+CASES+GAP_CASES+HEADER_CASES+GROUPED_DEVICE_CASES+GROUPED_FOLLOWUP_CASES+KV_OWNER_FOLLOWUP_CASES+KV_OWNER_REMAINING_CASES+SHARED_FOLLOWUP_CASES+QKV_PAIR_FOLLOWUP_CASES+ROW_CACHE_FOLLOWUP_CASES+PAIRED_GELU_FOLLOWUP_CASES+PAIRED_GELU_W8_FOLLOWUP_CASES+KV_OWNER_TAIL_CASES+SHARED_GELU_FOLLOWUP_CASES+KV_OWNER_W8_LARGE_CASES+QKV_REMAINING_CASES)
    parser.add_argument('--eager-only',action='store_true')
    parser.add_argument('--lookup-only',action='store_true',help='Only real production-cache rejections; no Graph/eager computation')
    parser.add_argument('--expected-kernels',type=Path)
    parser.add_argument('--dump-all-tensors',action='store_true',help='Save complete Graph/eager/recovery outputs and CPU references for offline review')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    expectations=json.loads(args.expected_kernels.read_text()) if args.expected_kernels else {}
    for name,env in [('rhinoOpLib_current.ref','RPU_KERNEL_LIB_PATH'),
                     ('rhinoExpansionOpLib_current.ref','RPU_KERNEL_LIB_PATH_EXPANSION')]:
        if name in expectations.get('expected_ref_sha256',{}):
            assert hashlib.sha256(Path(os.environ[env]).read_bytes()).hexdigest()==expectations['expected_ref_sha256'][name], 'unexpected production REF: '+name
    torch.set_num_threads(4);torch.ops.load_library(str(args.conformance))
    report=dict(model_execution=False,production_so=os.environ['RPU_BACKEND_SO'],
                production_sha256=hashlib.sha256(Path(os.environ['RPU_BACKEND_SO']).read_bytes()).hexdigest(),cases=[])
    for name in ([] if args.eager_only or args.lookup_only else args.case or CASES):
        rec=dict(case=name,status='RUNNING');report['cases'].append(rec)
        print('START '+name,flush=True)
        try:
            profile,rows,cores,mode,owners,images,indices,expected,linear=fixture(name)
            if name in PAIRED_LINEAR_REMAINING_CASES:
                pid=int(name.rsplit('p',1)[1]);spec=PAIRED_LINEAR_REMAINING_PROFILES[pid]
                rec['public_paired_linear_down_call']=dict(profile_id=pid,rows_per_input=spec[0],
                    input_count=1 if pid==120 else 2,output_count=2,local_n=2048,local_k=spec[2],
                    weight_dtype=str(owners[0].dtype),accumulator_bits=16,partition=0,has_bias=False,
                    contiguous_input_owners=pid==120,per_core_D16_last_k_probes=True,output_images_initially_nan=True)
            if name in PAIRED_LINEAR_ROW_CASES:
                rec['public_paired_linear_row_call']=dict(profile_id=int(name.rsplit('p',1)[1]),rows_per_input=rows,
                    input_count=2,output_count=2,local_n=2048,local_k=256,weight_dtype=str(owners[0].dtype),
                    accumulator_bits=16,partition=0,has_bias=False,signed_channel_scales=owners[0].dtype==torch.int8,
                    per_core_D16_last_k_probes=True,output_images_initially_nan=True)
            if name in QKV_REMAINING_CASES:
                qkv_id=int(name.rsplit('p',1)[1]);qkv_spec=QKV_REMAINING_PROFILES[qkv_id]
                rec['public_qkv_remaining_call']=dict(profile_id=qkv_id,rows=rows,local_k=qkv_spec[2],
                    local_ns=list(qkv_spec[3]),weight_dtype=str(owners[0].dtype),input_count=1,output_count=3,
                    signed_channel_scales=owners[0].dtype==torch.int8,all_core_D16_last_k_probes=True,
                    output_images_initially_nan=True,kv_tile_owner=False,acc32=False)
            if name in KV_OWNER_TAIL_CASES+KV_OWNER_W8_LARGE_CASES+SHARED_GELU_FOLLOWUP_CASES:
                rec['public_tail_followup_call']=dict(profile_id=int(name.rsplit('p',1)[1]),rows=rows,
                    local_n=32 if name in KV_OWNER_TAIL_CASES+KV_OWNER_W8_LARGE_CASES else 2048,local_k=2048,
                    weight_dtype=str(owners[0].dtype),signed_channel_scales=owners[0].dtype==torch.int8,
                    per_core_last_k_probes=True,output_images_initially_nan=True)
            if name in PAIRED_GELU_W8_FOLLOWUP_CASES:
                rec['public_paired_geglu_w8_call']=dict(profile_id=int(name.rsplit('p',1)[1]),rows_per_input=rows,
                    input_count=2,output_count=2,local_n=2048,local_k=2048,weight_dtype=str(owners[0].dtype),
                    accumulator_bits=16,signed_channel_scales=True,distinct_gate_up_scale_owners=True,
                    per_core_last_k_probes=True,output_images_initially_nan=True)
            if name in PAIRED_GELU_FOLLOWUP_CASES:
                rec['public_paired_geglu_call']=dict(profile_id=int(name.rsplit('p',1)[1]),rows_per_input=rows,
                    input_count=2,output_count=2,local_n=2048,local_k=2048,weight_dtype=str(owners[0].dtype),
                    accumulator_bits=16,per_core_last_k_probes=True,output_images_initially_nan=True)
            if name in SHARED_FOLLOWUP_CASES:
                profile_id=int(name.rsplit('p',1)[1]);spec=SHARED_FOLLOWUP_PROFILES[profile_id]
                rec['public_shared_call']=dict(profile_id=profile_id,rows=rows,local_n=spec[1],local_k=spec[2],
                    precision=spec[3],output_mode=spec[5],accumulator_bits=16)
            if name in QKV_PAIR_FOLLOWUP_CASES:
                rec['public_qkv_pair_call']=dict(profile_id=int(name.rsplit('p',1)[1]),rows=rows,
                    local_ns=[256,128,128],k=2048,weight_dtype=str(owners[0].dtype),
                    kv_tile_owner=bool(mode),compact_owner_width=64 if mode else None,acc32=False)
            if name in ROW_CACHE_FOLLOWUP_CASES:
                rec['public_row_cache_call']=dict(profile_id=int(name.rsplit('p',1)[1]),rows=rows,
                    n=owners[0].shape[0],global_k=owners[0].shape[1],local_k=owners[0].shape[1]//cores,
                    weight_dtype=str(owners[0].dtype),signed_scales=True,acc32=False,
                    causal_no_bias=profile=='causal_o_row_followup_w8')
            if name in GROUPED_DEVICE_CASES+GROUPED_FOLLOWUP_CASES:
                rec['expert_counts']=[im[1][0].contiguous().view(torch.int16)[:3].tolist() for im in images]
                rec['packed_assignment_rows']=rows
                rec['partition']='col' if mode==0 else 'row'
                rec['routing_varies_during_retained_replay']=rec['expert_counts'][0]!=rec['expert_counts'][1]
                if name in GROUPED_FOLLOWUP_CASES:
                    variant=name.removeprefix('generic_grouped_followup_').rsplit('_m',1)[0]
                    spec=GROUPED_FOLLOWUP_PROFILES[variant]
                    rec['public_grouped_call']=dict(weight_mode=spec['mode'],x_dtype=2,
                        acc32=spec['acc32'],local_n=spec['n'],local_k=spec['k'])
                if rows in (48,80,96):
                    rec['invalid_assignment_rows_rejected']={}
                    for invalid_rows in (0,65536):
                        try: torch.ops.rpu.test_production_linear_launchers(profile,owners,*images,indices,invalid_rows,cores,mode)
                        except (RuntimeError, ValueError) as error:
                            assert 'packed activation geometry' in str(error)
                            rec['invalid_assignment_rows_rejected'][str(invalid_rows)]=str(error)
                        else: raise AssertionError('invalid packed assignment rows accepted: '+str(invalid_rows))
            print('PHASE '+name+' fixture ready',flush=True)
            if name in KV_OWNER_REMAINING_CASES:
                rec['public_kv_owner_call']=dict(profile=int(name.rsplit('p',1)[1]),rows=rows,
                    local_n=32,k=2048,weight_dtype=str(owners[0].dtype),
                    acc32=False,signed_scales=owners[0].dtype==torch.int8,outputs_per_core=2)
            if name=='qwen_qkv_planar':
                try: torch.ops.rpu.test_production_linear_launchers(profile,owners,*images,indices,rows-16,cores,mode)
                except RuntimeError as error:
                    assert 'exact M192' in str(error),'wrong QKV envelope rejection'
                    rec['invalid_rows_rejected']=str(error)
                else: raise AssertionError('planar QKV accepted M176')
            if name in ('pi_w4_geglu_g32','pi_a8_split'):
                print('PHASE '+name+' negative owner',flush=True)
                invalid=list(owners)
                invalid[2]=owners[2].reshape(16,-1) if name=='pi_w4_geglu_g32' else owners[2].cpu().half().to('rpu')
                try:
                    torch.ops.rpu.test_production_linear_launchers(profile,invalid,*images,indices,rows,cores,mode)
                except RuntimeError as error:
                    required='GS32/64/128' if name=='pi_w4_geglu_g32' else 'FP32'
                    assert required in str(error),'negative case rejected at an unexpected boundary'
                    rec['invalid_scale_owner_rejected']=str(error)
                else: raise AssertionError('invalid production scale ABI accepted')
            owner_before=[owner.cpu().clone() for owner in owners]
            print('PHASE '+name+' valid production BUILD/REPLAY',flush=True)
            values,nodes,kargs,metadata=torch.ops.rpu.test_production_linear_launchers(profile,owners,*images,indices,rows,cores,mode)
            print('PHASE '+name+' production returned',flush=True)
            # Preserve actual dispatch and outputs even when a following
            # independent oracle check fails. The raw failure remains reviewable.
            (args.out/(name+'.nodes.jsonl')).write_text(nodes)
            (args.out/(name+'.args.jsonl')).write_text(kargs)
            if name in GROUPED_FOLLOWUP_CASES:
                rec['native_raw_parameter_states']=[json.loads(line).get('raw_state')
                    for line in kargs.splitlines() if line]
            if args.dump_all_tensors or name in PAIRED_LINEAR_REMAINING_CASES+PAIRED_LINEAR_ROW_CASES or name.startswith('pi_prefill_kv_owner') or name in QKV_PAIR_FOLLOWUP_CASES+ROW_CACHE_FOLLOWUP_CASES+SHARED_FOLLOWUP_CASES+PAIRED_GELU_FOLLOWUP_CASES+PAIRED_GELU_W8_FOLLOWUP_CASES+SHARED_GELU_FOLLOWUP_CASES+QKV_REMAINING_CASES:
                torch.save(dict(values=values,expected=expected,metadata=json.loads(metadata)),
                           args.out/(name+'.tensors.pt'))
            for owner_index,(owner,before) in enumerate(zip(owners,owner_before)):
                if profile=='pi_q_kv1_pair_cache_sspm' and owner_index in (6,7):
                    continue  # Full cache byte snapshots are exact outputs below.
                assert torch.equal(owner.cpu(),before),'read-only DDR owner changed'
            count=len(indices)+(2 if profile=='pi_q_kv1_pair_cache_sspm' else 0)
            parsed_nodes=[json.loads(line) for line in nodes.splitlines() if line]
            kernel_nodes=[node for node in parsed_nodes if 'grid' in node]
            assert kernel_nodes,'production Graph contains no kernel nodes'
            if name in expectations.get('cases',{}):
                actual=[dict(kernel=n['kernel'],grid=n['grid']) for n in kernel_nodes]
                assert actual==expectations['cases'][name], 'unexpected final production kernel/grid: '+str(actual)
            for node in kernel_nodes:
                assert len(node['grid'])==3 and all(v>0 for v in node['grid']),'invalid actual grid'
                assert node['cores']==list(range(cores)),'actual core broadcast differs'
            assert len(values)==4*count
            for i in range(2*count): assert torch.equal(values[i],values[i+2*count]),'same-input REPLAY differs'
            assert any(not torch.equal(values[i],values[i+count]) for i in range(count)),'A/B distinct inputs did not change outputs'
            metrics=[];exact_cpu_outputs=[]
            for arm in range(2):
                for j,reference in enumerate(expected[arm]):
                    actual=values[arm*count+j].view(reference.dtype).reshape_as(reference)
                    if profile=='pi_q_kv1_pair_cache_sspm' and j>=3:
                        assert actual.dtype==reference.dtype==torch.uint8
                        assert torch.equal(actual,reference),'full KV cache bytes differ (including untouched/padded regions)'
                        metrics.append(dict(arm=arm,output=j,full_cache_bytes_exact=True,bytes=actual.numel()))
                        if arm==0:exact_cpu_outputs.append(j)
                        continue
                    assert torch.isfinite(actual).all(),'nonfinite output'
                    assert torch.equal(actual.sign(),reference.sign()),'sparse semantic sign/zero pattern differs'
                    if profile in ('qwen_o_norm','wall_action_ring','hyvla_header_ring_fp16','hyvla_header_ring_w8') and j==0:
                        torch.testing.assert_close(actual,reference,rtol=0,atol=0)
                    if profile=='lingbot_router':
                        if j==0:torch.testing.assert_close(actual,reference,rtol=0,atol=0)
                        else:assert torch.equal(actual[:,rows:],torch.ones_like(actual[:,rows:])), 'router padded scores differ from +1 contract'
                    metrics.append(dict(arm=arm,output=j,**diagnostics(actual,reference)))
                    if linear and profile!='lingbot_router':
                        torch.testing.assert_close(actual,reference,rtol=0,atol=0)
                    if profile=='wall_base_shared_silu' and j==1:
                        torch.testing.assert_close(actual,reference,rtol=0,atol=0)
                    if arm==0 and ((linear and profile!='lingbot_router') or
                            (profile=='wall_base_shared_silu' and j==1) or
                            (profile in ('qwen_o_norm','wall_action_ring','lingbot_router','hyvla_header_ring_fp16','hyvla_header_ring_w8') and j==0)):
                        exact_cpu_outputs.append(j)
            (args.out/(name+'.nodes.jsonl')).write_text(nodes)
            (args.out/(name+'.args.jsonl')).write_text(kargs)
            rec.update(status='PASS',pass_scope='runtime/layout/replay and independent sparse semantic structure; floating errors are diagnostics',profile=profile,rows=rows,cores=cores,mode=mode,
                       independent_up_bias_zero_projection_probe=profile in ('wall_vision_triplet','wall_vision_shared_bias'),
                       exact_cpu_outputs=exact_cpu_outputs,
                       exact_sparse_linear=linear and profile!='lingbot_router',metrics=metrics,graph=json.loads(metadata),actual_kernel_nodes=kernel_nodes,owners=[dict(shape=list(owner.shape),dtype=str(owner.dtype),bytes=owner.numel()*owner.element_size()) for owner in owners])
        except Exception:
            rec.update(status='FAIL',error=traceback.format_exc())
        (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(rec),flush=True)
        if rec['status']=='FAIL': raise RuntimeError('case failed: '+name)
    if args.eager_only or args.lookup_only:
        # Initialize the real runtime before probing its cache; no kernel is
        # submitted by allocation or by the three rejected lookups.
        runtime_owner=torch.empty(1,device='rpu')
        report['kernel_lookup_rejections']=json.loads(torch.ops.rpu.test_production_kernel_lookup_rejections())
    if args.eager_only:
        report['eager_w4']=eager_w4_cases(args.out if args.dump_all_tensors else None)
        report['negative_owners']=negative_owner_cases(args.out if args.dump_all_tensors else None)
        print(json.dumps(dict(eager_w4=report['eager_w4'],negative_owners=report['negative_owners'])),flush=True)
    report['completed']=True
    (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
