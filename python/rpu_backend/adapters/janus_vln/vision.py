"""Device vision paths; CPU retains image rearrangement and position preparation."""
import torch
from torch import nn
from torch.nn import functional as F

from .attention import RPUProjection, rpu_online_attention


class RPUPatchConv(nn.Module):
    """Nonoverlapping patch14 convolution expressed as one device projection."""

    def __init__(self, conv):
        super().__init__()
        if (conv.kernel_size != (14, 14) or conv.stride != (14, 14)
                or conv.padding != (0, 0) or conv.dilation != (1, 1) or conv.groups != 1):
            raise ValueError("only nonoverlapping patch14 convolution is supported")
        linear = nn.Linear(conv.in_channels * 196, conv.out_channels,
                           bias=conv.bias is not None, device="meta")
        linear.weight = nn.Parameter(conv.weight.detach().flatten(1), requires_grad=False)
        if conv.bias is not None:
            linear.bias = nn.Parameter(conv.bias.detach(), requires_grad=False)
        self.projection = RPUProjection(linear)

    def forward(self, x):
        patches = F.unfold(x, kernel_size=14, stride=14).transpose(1, 2)
        result = self.projection(patches)
        return result.transpose(1, 2).reshape(x.shape[0], -1, x.shape[-2] // 14, x.shape[-1] // 14)


def install_rpu_linears(module):
    for name, child in list(module.named_children()):
        if isinstance(child, RPUProjection):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, RPUProjection(child))
        else:
            install_rpu_linears(child)


class RPUVisionPatch(nn.Module):
    def __init__(self, patch):
        super().__init__()
        conv = patch.proj
        if conv.kernel_size != (2, 14, 14) or conv.stride != (2, 14, 14) or conv.bias is not None:
            raise ValueError("expected Janus Qwen patch convolution")
        linear = nn.Linear(1176, conv.out_channels, bias=False, device="meta")
        linear.weight = nn.Parameter(conv.weight.detach().flatten(1), requires_grad=False)
        self.projection = RPUProjection(linear)

    def forward(self, pixels):
        return self.projection(pixels)


class RPUVisionAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        if original.num_heads != 16 or original.qkv.in_features != 1280:
            raise ValueError("expected Qwen vision H1280/16-head attention")
        self.qkv = RPUProjection(original.qkv)
        self.proj = RPUProjection(original.proj)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
        length = hidden_states.shape[0]
        if (cu_seqlens.ndim != 1 or cu_seqlens[0] != 0 or cu_seqlens[-1] != length
                or not torch.all(cu_seqlens[1:] > cu_seqlens[:-1])):
            raise ValueError("invalid Qwen image/window boundaries")
        q, k, v = self.qkv(hidden_states).reshape(length, 3, 16, 80).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())
        cos, sin = (t.unsqueeze(-2).float() for t in position_embeddings)
        def rotate(x):
            a, b = x.float().chunk(2, dim=-1)
            return (x.float() * cos + torch.cat((-b, a), dim=-1) * sin).half()
        q, k = rotate(q), rotate(k)
        output = torch.empty_like(v)
        for begin, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            result = rpu_online_attention(*(t[begin:end].transpose(0, 1)[None] for t in (q, k, v)))
            output[begin:end] = result[0].transpose(0, 1)
        return self.proj(output.reshape(length, 1280))
