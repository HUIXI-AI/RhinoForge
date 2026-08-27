"""Dual cache for Qwen3.5 hybrid attention.

Full-attention layers use the standard 7-D swizzled KV cache (`RPUCache`); GDN
layers carry a recurrent state + a causal-conv1d window state. All three are
parallel-indexed by layer (length == num_layers); full-attn slots of the GDN
states and GDN slots of the KV cache hold empty placeholders.

- `recurrent_states[L]`: `[num_v_heads, head_k_dim, head_v_dim]` — the gated
  delta-rule state (decode updates it in place each step).
- `conv_states[L]`: `[conv_kernel_dim, conv_dim]` — the conv1d window; slots
  0..k-2 are the previous inputs, slot k-1 is scratch (the decode kernel writes
  the new input there, then the conv output). conv output is read from slot k-1
  right after the step.

`gdn_states` is kept as an alias of `recurrent_states` for the C++ forward
signature.
"""
from __future__ import annotations

from numbers import Integral

import torch

from rpu_backend.api.cache import RPUCache


_VALID_LAYER_TYPES = frozenset({"linear_attention", "full_attention"})
QWEN3_5_OPTIONAL_PADDING_CAP = 64
QWEN3_5_TOTAL_PADDING_CAP = 63 + QWEN3_5_OPTIONAL_PADDING_CAP


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(
            f"Qwen3_5Cache {name} must be a positive integer, got {value!r}"
        )
    return int(value)


class Qwen3_5Cache(RPUCache):
    """RPUCache (KV for full layers) + per-GDN-layer recurrent_state + conv_state."""

    def __init__(self, num_layers, batch_size, max_seq_len,
                 num_kv_heads, head_dim, layer_types,
                 num_v_heads, key_head_dim, value_head_dim, conv_dim, conv_kernel_dim,
                 attn_tp=8, device="rpu", dtype=torch.float16):
        num_layers = _positive_int("num_layers", num_layers)
        logical_max_seq_len = _positive_int("max_seq_len", max_seq_len)
        num_v_heads = _positive_int("num_v_heads", num_v_heads)
        key_head_dim = _positive_int("key_head_dim", key_head_dim)
        value_head_dim = _positive_int("value_head_dim", value_head_dim)
        conv_dim = _positive_int("conv_dim", conv_dim)
        conv_kernel_dim = _positive_int("conv_kernel_dim", conv_kernel_dim)
        try:
            layer_types = tuple(layer_types)
        except TypeError as exc:
            raise ValueError("Qwen3_5Cache layer_types must be an iterable") from exc
        if len(layer_types) != num_layers:
            raise ValueError(
                "Qwen3_5Cache layer_types length must equal num_layers, got "
                f"{len(layer_types)} versus {num_layers}"
            )
        if any(not isinstance(layer_type, str) for layer_type in layer_types):
            raise ValueError("Qwen3_5Cache layer_types entries must be strings")
        invalid_types = sorted(set(layer_types) - _VALID_LAYER_TYPES)
        if invalid_types:
            raise ValueError(
                f"Qwen3_5Cache unsupported layer_types {invalid_types}; "
                f"expected only {sorted(_VALID_LAYER_TYPES)}"
            )
        num_cores = 8
        if conv_dim % num_cores != 0:
            raise ValueError(
                f"Qwen3_5Cache conv_dim must be divisible by 8, got {conv_dim}"
            )
        # Route-B first adds up to 63 mandatory rows for the 64-row GDN grid,
        # then may spend the public optional budget. Keep the public horizon
        # logical while preserving the complete configured search window even
        # when the prompt consumes that horizon.
        physical_max_seq_len = (
            (logical_max_seq_len + QWEN3_5_TOTAL_PADDING_CAP + 15) // 16
        ) * 16
        self.allocated_max_seq_len = physical_max_seq_len
        # The factory below expands logical full-attention KV heads to an
        # effective 8-head width and passes attn_tp=8, matching the SDPA
        # virtual-8 layout used by the swizzled 7-D cache.
        super().__init__(num_layers, batch_size, physical_max_seq_len,
                         num_kv_heads, head_dim, attn_tp=attn_tp,
                         device=device, dtype=dtype)
        self.max_seq_len = logical_max_seq_len
        empty = torch.empty(0, device=device, dtype=dtype)

        def _is_gdn(t):
            return t == "linear_attention"

        self.recurrent_states = [
            torch.zeros(num_v_heads, key_head_dim, value_head_dim,
                        device=device, dtype=dtype) if _is_gdn(t) else empty
            for t in layer_types]
        # conv_state core-major [num_cores, Kc, conv_dim/num_cores] so each core's
        # conv window is a CONTIGUOUS slice (scatter-friendly) — the fused decode
        # runs the GDN mixer 2-heads/core (8-core), and the per-core channels are
        # the reordered [2q|2k|2v] block (see adapter weight prep). conv_dim must
        # be divisible by num_cores.
        self.conv_states = [
            torch.zeros(num_cores, conv_kernel_dim, conv_dim // num_cores,
                        device=device, dtype=dtype)
            if _is_gdn(t) else empty
            for t in layer_types]
        # C++ forward takes `gdn_states` (== recurrent_states).
        self.gdn_states = self.recurrent_states

        # DDR optimization: only full-attention layers use the 7-D KV cache; GDN
        # (`linear_attention`) layers carry recurrent+conv state and NEVER touch KV.
        # The fused forward dereferences k_caches[layer_idx] only for a full
        # attention layer, so an empty tensor at a GDN index is never indexed.
        # Conversely, gdn_states/conv_states are empty at full-attention layers.
        # The base RPUCache
        # allocated all 32 layers above; drop the ~75% (4B: 24/32) that are GDN.
        # Cost: super().__init__ momentarily peaks at the full allocation before
        # these are freed (fine on a large-DDR board; switch to lazy alloc only if
        # that transient OOMs). reset() stays correct: empty.zero_() is a no-op.
        for i, t in enumerate(layer_types):
            if _is_gdn(t):
                self.k_caches[i] = empty
                self.v_caches[i] = empty

    def reset(self):
        """Reset to an empty conversation: KV cache + position (super) plus the
        per-GDN-layer recurrent + conv-window states. Zeroed IN PLACE so the DDR
        addresses the fused decode graph baked at BUILD stay valid — callers that
        reuse one cache across turns must reset rather than reallocate."""
        super().reset()
        for s in self.recurrent_states:
            s.zero_()
        for s in self.conv_states:
            s.zero_()

    def reset_to_position(self, pos: int):
        """Reset fully at zero; reject non-zero rewinds of recurrent GDN state."""
        if isinstance(pos, bool) or not isinstance(pos, Integral):
            raise ValueError(
                f"Qwen3_5Cache position must be an integer, got {pos!r}"
            )
        pos = int(pos)
        if pos == 0:
            self.reset()
            return
        if pos == self.position:
            return
        raise ValueError(
            "Qwen3_5Cache cannot reset to a non-zero position: KV slots can be "
            "rewound, but recurrent/conv GDN state cannot be reconstructed. "
            "Call reset_to_position(0) and re-feed the full prefix instead."
        )

    @classmethod
    def from_config(cls, text_config, max_seq_len, device="rpu",
                    dtype=torch.float16):
        NUM_CORES = 8
        max_seq_len = _positive_int("max_seq_len", max_seq_len)
        nkv = _positive_int(
            "num_key_value_heads", text_config.num_key_value_heads
        )
        # Full-attention KV heads are replicated to ``nkv_eff == NUM_CORES``
        # (one complete KV head per core). GDN states are unaffected.
        kv_rep = NUM_CORES // nkv if (nkv < NUM_CORES and NUM_CORES % nkv == 0) else 1
        nkv_eff = nkv * kv_rep
        attn_tp = NUM_CORES             # full-attn now runs on all 8 cores (replicated KV)
        nvh = _positive_int(
            "linear_num_value_heads", text_config.linear_num_value_heads
        )
        nkh = _positive_int(
            "linear_num_key_heads", text_config.linear_num_key_heads
        )
        kdim = _positive_int(
            "linear_key_head_dim", text_config.linear_key_head_dim
        )
        vdim = _positive_int(
            "linear_value_head_dim", text_config.linear_value_head_dim
        )
        # No GQA weight replication: q/k stay at nkh heads (the nkh→nvh expansion is
        # a runtime tile in build_gdn_mixer). conv_dim uses TRUE nkh — matches
        # the text adapter's ``gdn_conv_dim`` and the native mixer (8192 for 4B).
        conv_dim = 2 * (nkh * kdim) + nvh * vdim
        return cls(
            num_layers=_positive_int(
                "num_hidden_layers", text_config.num_hidden_layers
            ),
            batch_size=1,
            max_seq_len=max_seq_len,
            num_kv_heads=nkv_eff,
            head_dim=_positive_int("head_dim", text_config.head_dim),
            layer_types=text_config.layer_types,
            attn_tp=attn_tp,
            num_v_heads=nvh,
            key_head_dim=kdim,
            value_head_dim=vdim,
            conv_dim=conv_dim,
            conv_kernel_dim=_positive_int(
                "linear_conv_kernel_dim", text_config.linear_conv_kernel_dim
            ),
            device=device, dtype=dtype,
        )
