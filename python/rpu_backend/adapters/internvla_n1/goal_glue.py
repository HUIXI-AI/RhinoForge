"""InternVLA-N1 CPU-fp32 goal glue: vlm_tokens[1,4,3584] → goal_embed[1,1,384].

Bridges System-2 (vlm_tokens) to System-1 (NavDP memory). Two small modules:
  * vlm_embed_mlp: 3584 → 896 → 448 → 384 (Linear/ReLU × 2 + Linear)
  * goal_compressor (TokenCompressor): a 1-query cross-attention (8 heads) that compresses the
    4 vlm tokens to a single goal token, with learned token/query positional encodings.

Reconstructed as plain nn.Modules (no InternNav import); weights load strictly from the
extracted `navdp_head.safetensors` (`vlm_embed_mlp.*`, `goal_compressor.*`). The unvalidated
stock-aten RPU path is deliberately rejected.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        b, s, _ = x.shape
        pid = torch.arange(s, dtype=torch.long, device=x.device).unsqueeze(0).expand(b, -1)
        return self.position_embedding(pid)


class TokenCompressor(nn.Module):
    def __init__(self, embed_dim, num_heads, target_length):
        super().__init__()
        self.target_embedding = nn.Embedding(target_length, embed_dim)
        self.token_positional_encoding = LearnablePositionalEncoding(embed_dim)
        self.query_positional_encoding = LearnablePositionalEncoding(embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.embed_dim, self.num_heads = embed_dim, num_heads

    def forward(self, x, padding_mask=None):
        # Manual MHA (avoids nn.MultiheadAttention's internal views, which alias on the RPU
        # custom device — "View returned a tensor same as base"). Numerically identical: same
        # in_proj_weight/bias + out_proj, F.scaled_dot_product_attention. padding_mask unused
        # (goal_compressor is always called with the full 4 vlm tokens, no padding).
        bs, E, H = x.size(0), self.embed_dim, self.num_heads
        hd = E // H
        x = x + self.token_positional_encoding(x)
        q = self.target_embedding.weight.unsqueeze(0).repeat(bs, 1, 1)   # [bs,1,E]
        q = q + self.query_positional_encoding(q)
        w, b = self.cross_attention.in_proj_weight, self.cross_attention.in_proj_bias
        qh = F.linear(q, w[:E], b[:E]).reshape(bs, -1, H, hd).transpose(1, 2).contiguous()
        kh = F.linear(x, w[E:2 * E], b[E:2 * E]).reshape(bs, -1, H, hd).transpose(1, 2).contiguous()
        vh = F.linear(x, w[2 * E:], b[2 * E:]).reshape(bs, -1, H, hd).transpose(1, 2).contiguous()
        o = F.scaled_dot_product_attention(qh, kh, vh)                   # [bs,H,1,hd]
        o = o.transpose(1, 2).contiguous().reshape(bs, -1, E)           # [bs,1,E]
        return self.cross_attention.out_proj(o)


class GoalGlue(nn.Module):
    """vlm_tokens[bs,4,3584] → goal_embed[bs,1,384]."""

    def __init__(self, vlm_dim=3584, token_dim=384):
        super().__init__()
        self.vlm_embed_mlp = nn.Sequential(
            nn.Linear(vlm_dim, vlm_dim // 4), nn.ReLU(),
            nn.Linear(vlm_dim // 4, vlm_dim // 8), nn.ReLU(),
            nn.Linear(vlm_dim // 8, token_dim))
        self.goal_compressor = TokenCompressor(token_dim, 8, 1)

    def forward(self, vlm_tokens):
        return self.goal_compressor(self.vlm_embed_mlp(vlm_tokens))


def build_goal_glue(safetensors_path: str | Path, device: str = "cpu") -> GoalGlue:
    """Load the CPU-fp32 goal glue from ``navdp_head.safetensors``."""
    if device != "cpu":
        raise ValueError("InternVLA-N1 goal glue supports only device='cpu' with fp32")
    from safetensors.torch import load_file
    sd = load_file(str(safetensors_path))
    g = GoalGlue().eval()
    g.vlm_embed_mlp.load_state_dict({
        k[len("vlm_embed_mlp."):]: v.float()
        for k, v in sd.items() if k.startswith("vlm_embed_mlp.")
    }, strict=True)
    compressor_state = {
        k[len("goal_compressor."):]: v.float()
        for k, v in sd.items() if k.startswith("goal_compressor.")
    }
    legacy_pe = compressor_state.pop("positional_encoding.pe", None)
    if legacy_pe is not None and tuple(legacy_pe.shape) != (1000, g.goal_compressor.embed_dim):
        raise ValueError(
            "InternVLA-N1 legacy positional_encoding.pe must have shape "
            f"(1000, {g.goal_compressor.embed_dim}), got {tuple(legacy_pe.shape)}"
        )
    g.goal_compressor.load_state_dict(compressor_state, strict=True)
    return g.float().cpu()
