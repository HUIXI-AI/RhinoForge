"""Shared CPU suffix construction and the two checkpoint-defined flow schedules."""

from numbers import Integral
import struct

import torch


def denoise_schedule(steps, flow_direction="legacy_ascending"):
    """Preserve upstream float32 recurrence, including its rounding at each step."""
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0:
        raise ValueError("RhinoVLA denoise steps must be a positive integer")
    if flow_direction not in ("legacy_ascending", "official_descending"):
        raise ValueError("unknown RhinoVLA checkpoint flow direction")
    descending = flow_direction == "official_descending"
    dt = (-1.0 if descending else 1.0) / steps
    time = torch.tensor(1.0 if descending else 0.0, dtype=torch.float32)
    times = []
    for _ in range(steps):
        times.append(float(time))
        time = (time + dt).clamp(min=0.0) if descending else (time + dt).clamp(max=1.0)
    # Exact schedule identity for graph admission; all words fit signed int64.
    words = (int(steps), *struct.unpack("<II", struct.pack("<d", dt)),
             *(struct.unpack("<I", struct.pack("<f", t))[0] for t in times))
    return tuple(times), dt, words


def build_suffix(action_io, mask_proj, state, x_t, time, state_mask, action_mask):
    """Apply the source model's mask conditioning after its own embed_suffix."""
    suffix_embeds, suffix_mask, adarms_cond = action_io.embed_suffix(state, x_t, time)
    if mask_proj is not None:
        if action_mask is None:
            raise ValueError("RhinoVLA mask conditioning requires action_mask")
        action_start = 1 if action_io.state_proj is not None else 0
        if action_io.state_proj is not None:
            if state_mask is None:
                raise ValueError("RhinoVLA state token conditioning requires state_mask")
            sm = state_mask[:, 0] if state_mask.ndim == 3 else state_mask
            suffix_embeds[:, :1] = suffix_embeds[:, :1] + mask_proj["state_mask_proj"](
                sm.to(suffix_embeds))[:, None, :]
        if action_mask.ndim == 2:
            action_mask = action_mask[:, None, :]
        action_tokens = suffix_embeds.shape[1] - action_start
        if action_mask.ndim != 3:
            raise ValueError("RhinoVLA action_mask must have rank 2 or 3")
        if action_mask.shape[1] == 1:
            action_mask = action_mask.expand(-1, action_tokens, -1)
        if action_mask.shape[:2] != (suffix_embeds.shape[0], action_tokens):
            raise ValueError("RhinoVLA action_mask tokens do not match suffix action tokens")
        suffix_embeds[:, action_start:] = suffix_embeds[:, action_start:] + mask_proj["action_mask_proj"](
            action_mask.to(suffix_embeds))
    return suffix_embeds, suffix_mask, adarms_cond


def suffix_position_ids(prefix_mask, suffix_mask, prefix_rope_deltas=None):
    prefix_off = prefix_mask.long().sum(-1)[:, None]
    if prefix_rope_deltas is not None:
        delta = prefix_rope_deltas
        if isinstance(delta, torch.Tensor):
            if delta.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                raise TypeError("RhinoVLA prefix_rope_deltas must be an integer tensor")
            if tuple(delta.shape) not in ((prefix_off.shape[0],), tuple(prefix_off.shape)):
                raise ValueError("RhinoVLA prefix_rope_deltas must have shape [B] or [B,1]")
            delta = delta.reshape_as(prefix_off).to(prefix_off)
        elif isinstance(delta, bool) or not isinstance(delta, Integral):
            raise TypeError("RhinoVLA prefix_rope_deltas must be integral")
        prefix_off = prefix_off + delta
    if bool((prefix_off < 0).any()):
        raise ValueError("RhinoVLA logical prefix + RoPE delta must be non-negative")
    suffix_pos = prefix_off + torch.cumsum(suffix_mask.long(), 1) - 1
    return suffix_pos[None, ...].expand(3, -1, -1)


def rpu_denoise(expert_rpu, action_io, mask_proj, *, prefix_key_values, prefix_mask,
                state, state_mask, action_mask, x0, steps=10, return_velocities=False,
                flow_direction="legacy_ascending", prefix_rope_deltas=None):
    """CPU suffix/decode around the patched expert, with checkpoint-defined Euler flow."""
    times, dt, _ = denoise_schedule(steps, flow_direction)
    descending = flow_direction == "official_descending"
    x_t = x0.clone()
    if action_mask is not None and action_mask.ndim == 2:
        action_mask = action_mask[:, None, :]
    if descending and action_mask is not None:
        x_t = x_t * action_mask
    if descending and state is not None and state_mask is not None:
        if state_mask.shape != state.shape:
            raise ValueError("RhinoVLA state_mask shape must match state")
        state = state * state_mask
    velocities = []
    for t in times:
        time = torch.full((x0.shape[0],), t, dtype=torch.float32, device=x0.device)
        suffix_embeds, suffix_mask, adarms_cond = build_suffix(
            action_io, mask_proj, state, x_t, time, state_mask, action_mask)
        position_ids = suffix_position_ids(prefix_mask, suffix_mask, prefix_rope_deltas)
        hidden = expert_rpu(suffix_embeds, prefix_key_values=prefix_key_values,
                            prefix_mask=prefix_mask, suffix_mask=suffix_mask,
                            position_ids=position_ids, adarms_cond=adarms_cond,
                            prefix_layer_offset=0, prefix_rope_deltas=prefix_rope_deltas)
        v = action_io.decode_actions(hidden.to("cpu").float())
        if not descending and action_mask is not None:
            v = v * action_mask
        if return_velocities:
            velocities.append(v.detach().clone())
        x_t = x_t + dt * v
        if action_mask is not None:
            x_t = x_t * action_mask
    return (x_t, velocities) if return_velocities else x_t


def build_rhino_inputs(cfg, prefix_len=48, batch_size=1, use_mask_condition=True, seed=42):
    """Synthetic prefix/state for the existing explicit skip-vision diagnostic."""
    torch.manual_seed(seed)
    prefix_kv = []
    for _ in range(cfg.depth):
        k = torch.randn(batch_size, cfg.num_key_value_heads, prefix_len, cfg.head_dim) * 0.1
        v = torch.randn(batch_size, cfg.num_key_value_heads, prefix_len, cfg.head_dim) * 0.1
        prefix_kv.append((k, v))
    prefix_mask = torch.ones(batch_size, prefix_len, dtype=torch.bool)
    state = torch.randn(batch_size, cfg.state_dim) * 0.1
    x_t = torch.randn(batch_size, cfg.action_horizon, cfg.action_dim) * 0.5
    state_mask = action_mask = None
    if use_mask_condition:
        state_mask = torch.ones(batch_size, cfg.state_dim)
        action_mask = torch.ones(batch_size, cfg.action_horizon, cfg.action_dim)
        state_mask[:, -8:] = 0
        action_mask[:, :, -8:] = 0
    return prefix_kv, prefix_mask, state, x_t, state_mask, action_mask
