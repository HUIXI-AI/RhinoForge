"""Gemma4 per-layer mixed-geometry table + graph-signature encoding.

Pure-Python by design: NO torch / transformers / rpu_backend imports at module
load, so this file is safe to import standalone (unit tests) AND from the
hardware-gated adapter ``__init__`` without dragging in the C++ extension. It is
the single source of truth for Gemma4's mixed geometry, consumed by both:

  * the C++ ``Gemma4Model`` ``set_weights`` per-layer int arrays
    (``layer_type_ids`` / ``head_dim_per_layer`` / ``num_kv_heads_per_layer`` /
    ``kv_source_layer`` / ``is_kv_shared``), and
  * the Python graph-signature key (``geometry_dyn_dims``) so a geometry change
    forces a fresh BUILD instead of a wrong REPLAY hit; the base graph
    signature intentionally omits position.

KV-share semantics: the source is the last non-shared layer of each type;
shared layers with ``idx >= L - num_kv_shared_layers`` drop k/v/k_norm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

GLOBAL_LAYER_TYPE = "full_attention"
SLIDING_LAYER_TYPE = "sliding_attention"


@dataclass(frozen=True)
class LayerGeom:
    """Geometry of one decoder layer (mixed sliding/global widths)."""

    idx: int
    layer_type: str            # "sliding_attention" | "full_attention"
    is_global: bool
    head_dim: int              # 256 (sliding) / 512 (global)
    num_q_heads: int           # 8
    num_kv_heads: int          # 2 (E4B; attention_k_eq_v=False)
    rope_type: str             # "default" (sliding) | "proportional" (global)
    sliding_window: Optional[int]
    is_kv_shared: bool         # idx >= first_shared  -> reuse source layer K/V
    kv_source_layer: int       # layer whose K/V slot SDPA reads (== idx if not shared)
    has_kv_proj: bool          # False for shared layers (HF drops k/v/k_norm)

    @property
    def layer_type_id(self) -> int:
        """0 = sliding, 1 = global (matches C++ layer_type_ids convention)."""
        return 1 if self.is_global else 0


def _cfg(tc: Any, name: str, default: Any = None) -> Any:
    """Read a field from a HF config object or a plain dict (test path)."""
    if isinstance(tc, dict):
        return tc.get(name, default)
    return getattr(tc, name, default)


def layer_geometry(text_config: Any) -> list[LayerGeom]:
    """Build the per-layer geometry table from a Gemma4 ``text_config``.

    ``text_config`` may be a HF ``Gemma4TextConfig`` or a plain dict (the JSON
    ``text_config`` block) — both are accepted so this is testable offline.
    """
    L = int(_cfg(text_config, "num_hidden_layers"))
    layer_types = list(_cfg(text_config, "layer_types"))
    if len(layer_types) != L:
        raise ValueError(
            f"layer_types length {len(layer_types)} != num_hidden_layers {L}")

    num_shared = int(_cfg(text_config, "num_kv_shared_layers", 0) or 0)
    first_shared = L - num_shared

    head_dim = int(_cfg(text_config, "head_dim"))
    global_head_dim = int(_cfg(text_config, "global_head_dim") or head_dim)
    nq = int(_cfg(text_config, "num_attention_heads"))
    nkv = int(_cfg(text_config, "num_key_value_heads"))
    nkv_global = _cfg(text_config, "num_global_key_value_heads")
    k_eq_v = bool(_cfg(text_config, "attention_k_eq_v", False))
    sliding_window = _cfg(text_config, "sliding_window")

    # Source layer per type = last non-shared layer of that type.
    last_nonshared: dict[str, int] = {}
    for i in range(first_shared):
        last_nonshared[layer_types[i]] = i

    geo: list[LayerGeom] = []
    for i in range(L):
        lt = layer_types[i]
        is_global = lt == GLOBAL_LAYER_TYPE
        hd = global_head_dim if is_global else head_dim
        use_alt_kv = k_eq_v and is_global
        this_nkv = int(nkv_global if (use_alt_kv and nkv_global) else nkv)
        is_shared = i >= first_shared
        if is_shared:
            if lt not in last_nonshared:
                raise ValueError(
                    f"layer {i} ({lt}) is KV-shared but no non-shared source "
                    f"layer of that type exists in [0,{first_shared})")
            src = last_nonshared[lt]
        else:
            src = i
        geo.append(LayerGeom(
            idx=i,
            layer_type=lt,
            is_global=is_global,
            head_dim=hd,
            num_q_heads=nq,
            num_kv_heads=this_nkv,
            rope_type="proportional" if is_global else "default",
            sliding_window=None if is_global else (
                int(sliding_window) if sliding_window else None),
            is_kv_shared=is_shared,
            kv_source_layer=src,
            has_kv_proj=not is_shared,
        ))
    return geo


def cpp_layer_arrays(geo: list[LayerGeom]) -> dict[str, list[int]]:
    """Per-layer int arrays passed verbatim to C++ ``gemma4_set_weights``."""
    return {
        "layer_type_ids":        [g.layer_type_id for g in geo],
        "head_dim_per_layer":    [g.head_dim for g in geo],
        "num_kv_heads_per_layer": [g.num_kv_heads for g in geo],
        "kv_source_layer":       [g.kv_source_layer for g in geo],
        "is_kv_shared":          [int(g.is_kv_shared) for g in geo],
        "has_kv_proj":           [int(g.has_kv_proj) for g in geo],
    }


def max_head_dim(geo: list[LayerGeom]) -> int:
    """Widest per-layer head_dim — buffers/cache slots are sized to this."""
    return max(g.head_dim for g in geo)


def max_num_kv_heads(geo: list[LayerGeom]) -> int:
    return max(g.num_kv_heads for g in geo)


# --- graph-signature geometry encoding --------------------------------------
_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_U64 = (1 << 64) - 1
_POS63 = (1 << 63) - 1  # keep the hash a non-negative int63 (safe for int dyn_dims)


def geometry_hash(geo: list[LayerGeom]) -> int:
    """Order-sensitive FNV-1a hash over the per-layer geometry tuples.

    Folds every dimension that changes the emitted graph (head_dim, num_kv_heads,
    sliding-vs-global, KV-share routing). Two models whose layer geometry differs
    in any of these yield different hashes -> different graph signatures.
    """
    h = _FNV64_OFFSET
    for g in geo:
        for v in (g.head_dim, g.num_kv_heads, g.layer_type_id,
                  int(g.is_kv_shared), g.kv_source_layer):
            h = ((h ^ (int(v) & 0xFFFFFFFF)) * _FNV64_PRIME) & _U64
    return h & _POS63


def geometry_dyn_dims(geo: list[LayerGeom]) -> list[int]:
    """Extra ``dyn_dims`` appended to the graph signature so a geometry change
    forces a fresh BUILD (the base sig drops position and only carries
    [seq_len, hidden] + [num_layers, ...]). Returns [num_layers, geometry_hash].
    """
    return [len(geo), geometry_hash(geo)]
