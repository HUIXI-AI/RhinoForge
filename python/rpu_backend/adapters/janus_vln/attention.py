"""Bounded host attention for the controlled Janus R1 hybrid port.

RPU owns FP16 projections. CPU owns normalization, RoPE and FP32 softmax.
This explicit route does not claim a native RPU SDPA kernel.
"""
import copy

import torch
from torch import nn


@torch.inference_mode()
def online_attention(q, k, v, *, query_chunk=128, key_chunk=1024):
    """Unmasked attention with one global softmax, bounded in both axes."""
    if (q.device.type != "cpu" or k.device.type != "cpu" or v.device.type != "cpu"
            or q.ndim != 4 or k.ndim != 4 or v.shape != k.shape
            or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]
            or k.shape[-2] == 0):
        raise ValueError("attention requires compatible nonempty CPU [B,H,S,D] tensors")
    if any(type(n) is not int or n < 1 for n in (query_chunk, key_chunk)):
        raise ValueError("chunk sizes must be positive integers")
    output = torch.empty_like(q, dtype=torch.float32)
    scale = q.shape[-1] ** -0.5
    for start in range(0, q.shape[-2], query_chunk):
        query = q[..., start:start + query_chunk, :].float()
        maximum = torch.full((*query.shape[:-1], 1), -torch.inf)
        denominator = torch.zeros_like(maximum)
        accumulator = torch.zeros_like(query)
        for offset in range(0, k.shape[-2], key_chunk):
            key = k[..., offset:offset + key_chunk, :].float()
            value = v[..., offset:offset + key_chunk, :].float()
            scores = (query @ key.transpose(-1, -2)) * scale
            new_maximum = torch.maximum(maximum, scores.amax(-1, keepdim=True))
            correction = (maximum - new_maximum).exp()
            scores.sub_(new_maximum).exp_()
            accumulator.mul_(correction).add_(scores @ value)
            denominator.mul_(correction).add_(scores.sum(-1, keepdim=True))
            maximum = new_maximum
        output[..., start:start + query_chunk, :] = accumulator / denominator
    return output


class HostAttention(nn.Module):
    """VGGT attention contract; raw host K/V remains owned by the caller."""

    def __init__(self, original):
        super().__init__()
        if (original.num_heads != 16 or original.head_dim != 64
                or original.qkv.in_features != 1024 or original.training):
            raise ValueError("expected evaluation-mode VGGT H1024/16-head attention")
        self.qkv = original.qkv
        self.proj = original.proj
        self.q_norm = copy.deepcopy(original.q_norm).cpu().float()
        self.k_norm = copy.deepcopy(original.k_norm).cpu().float()
        self.rope = copy.deepcopy(original.rope).cpu() if original.rope is not None else None

    @torch.inference_mode()
    def forward(self, x, pos=None, attn_mask=None, past_key_values=None, use_cache=False):
        if x.device.type != "cpu" or x.ndim != 3 or x.shape[0] != 1:
            raise ValueError("controlled VGGT attention requires CPU batch size one")
        if attn_mask is not None:
            raise ValueError("masked attention is outside this Janus profile")
        if past_key_values is not None and not use_cache:
            raise ValueError("past state requires use_cache=True")
        batch, tokens, channels = x.shape
        if channels != 1024 or not 1 <= tokens <= 1615:
            raise ValueError("VGGT input exceeds the fixed image envelope")
        q, k, v = self.qkv(x.half()).reshape(batch, tokens, 3, 16, 64).permute(2, 0, 3, 1, 4).unbind(0)
        new_kv = None
        if use_cache:
            k, v = k.unsqueeze(2), v.unsqueeze(2)
            if past_key_values is not None:
                if (len(past_key_values) != 2 or any(
                        t.device.type != "cpu" or t.dtype != torch.float16
                        or t.ndim != 5 or t.shape[:2] != (1, 16)
                        or t.shape[2] > 56 or t.shape[3:] != (tokens, 64)
                        for t in past_key_values)
                        or past_key_values[0].shape != past_key_values[1].shape):
                    raise ValueError("invalid raw VGGT cache")
                k = torch.cat((past_key_values[0], k), dim=2)
                v = torch.cat((past_key_values[1], v), dim=2)
            new_kv = (k, v)
            k, v = k.flatten(2, 3), v.flatten(2, 3)
        q = self.q_norm(q.float())
        # Only the active layer holds normalized keys; persistent cache stays raw FP16.
        output = torch.empty_like(k, dtype=torch.float32)
        for start in range(0, k.shape[-2], tokens):
            key = self.k_norm(k[..., start:start + tokens, :].float())
            if self.rope is not None:
                key = self.rope(key, pos)
            output[..., start:start + tokens, :] = key
        if self.rope is not None:
            q = self.rope(q, pos)
        result = online_attention(q, output, v).transpose(1, 2).reshape(batch, tokens, channels)
        result = self.proj(result.half())
        if use_cache and new_kv[0].shape[2] > 56:
            # Crop after attending to frame 57, one layer at a time.
            new_kv = tuple(torch.cat((t[:, :, :8], t[:, :, -48:]), dim=2) for t in new_kv)
        return (result, new_kv) if use_cache else result


class RPUProjection(nn.Module):
    """One-time swizzle, with bounded immediate projection launches."""

    def __init__(self, linear):
        super().__init__()
        from rpu_backend.runtime.weights import swizzle_model_inplace
        if linear.weight.device.type != "cpu" or hasattr(linear, "_rpu_linear_partition"):
            raise ValueError("projection requires fresh CPU row-major weights")
        self.linear = copy.deepcopy(linear).half()
        swizzle_model_inplace(self)
        self.linear.to("rpu")

    @torch.inference_mode()
    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1]).half()
        output = torch.empty((flat.shape[0], self.linear.out_features), dtype=torch.float16)
        for start in range(0, flat.shape[0], 64):
            output[start:start + 64] = self.linear(flat[start:start + 64].to("rpu")).cpu()
        return output.reshape(*x.shape[:-1], self.linear.out_features)


def adapt_attention(original):
    adapted = HostAttention(original)
    adapted.qkv = RPUProjection(original.qkv)
    adapted.proj = RPUProjection(original.proj)
    return adapted.eval()
