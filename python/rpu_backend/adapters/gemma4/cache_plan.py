"""Gemma4 mixed-width KV-cache plan and required source map.

Pure Python (stdlib only) so it is standalone-loadable for offline checks
(no torch / transformers / rpu_backend import). Computes, per decoder layer:
  * the 7-D swizzled K/V slot shapes for that layer's (num_kv_heads, head_dim), and
  * the forced source-map: which layer's slot each layer's SDPA reads.

Why a source map is mandatory: Gemma4 KV sharing (idx >= first
shared) means a layer must READ ANOTHER layer's K/V slot. A "max-width per-layer"
option alone does not express that, so the plan must always produce the source map. The
C++ ``Gemma4Model::build_layer_subgraph`` reads ``k_caches[geom.kv_source_layer]``
and skips k/v insert for shared layers, so ``Gemma4KVCache`` (cache.py) only has
to allocate a slot per non-shared layer and ALIAS shared layers to their source.

⚠️ The 7-D layout must stay in sync with
``rpu_backend/api/cache.py`` ``RPUCache.__init__`` (the insert_kcache/vcache +
attention operators assume it). It is mirrored rather than imported to keep
this module package-independent. Hardware constant NUM_CORES = 8.
"""
from __future__ import annotations

from typing import Any

# 7-D swizzle constants — mirror rpu_backend/api/cache.py RPUCache.
NUM_CORES = 8
S_KEY_CHUNK = 16
S_VAL_CHUNK = 16
HEAD_DIM_CHUNK = 16


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def kv_slot_shapes(num_kv_heads: int, head_dim: int, max_seq_len: int,
                   attn_tp: int, batch_size: int = 1,
                   num_cores: int = NUM_CORES) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """7-D (K, V) swizzle shapes for ONE layer's (num_kv_heads, head_dim).

    Mirrors RPUCache.__init__: nKVHeadChunk * nKVHeadVx == num_cores*nkv//attn_tp.
      K: [B, sKeyVx, nKVHeadVx, headDimVx, nKVHeadChunk, sKeyChunk, headDimChunk]
      V: [B, sValVx, nKVHeadVx, headDimVx, nKVHeadChunk, headDimChunk, sValChunk]
    """
    if head_dim % HEAD_DIM_CHUNK != 0:
        raise ValueError(f"head_dim {head_dim} not a multiple of {HEAD_DIM_CHUNK}")
    effective_kv_slots = num_cores * num_kv_heads // attn_tp
    n_kv_head_chunk = min(num_cores, effective_kv_slots)
    n_kv_head_vx = _ceil_div(effective_kv_slots, n_kv_head_chunk)
    head_dim_vx = head_dim // HEAD_DIM_CHUNK
    s_key_vx = _ceil_div(max_seq_len, S_KEY_CHUNK)
    s_val_vx = _ceil_div(max_seq_len, S_VAL_CHUNK)
    k_shape = (batch_size, s_key_vx, n_kv_head_vx, head_dim_vx,
               n_kv_head_chunk, S_KEY_CHUNK, HEAD_DIM_CHUNK)
    v_shape = (batch_size, s_val_vx, n_kv_head_vx, head_dim_vx,
               n_kv_head_chunk, HEAD_DIM_CHUNK, S_VAL_CHUNK)
    return k_shape, v_shape


def cache_plan(geom: list[Any], max_seq_len: int, attn_tp: int,
               batch_size: int = 1, num_cores: int = NUM_CORES) -> dict:
    """Per-layer KV-cache plan for a Gemma4 geometry table.

    ``geom`` is a list of LayerGeom-like objects (duck-typed: ``.head_dim``,
    ``.num_kv_heads``, ``.kv_source_layer``, ``.is_kv_shared``) — passed by value
    so this stays rpu_backend-free.

    Returns a dict with:
      * ``num_layers``
      * ``source_map``   : source_map[i] = layer whose slot layer i's SDPA reads
                           (== i when layer i owns its slot)
      * ``owners``       : sorted layer indices that ALLOCATE a distinct slot
                           (the non-shared layers)
      * ``layers``       : per-layer dicts {idx, owns, source, head_dim,
                           num_kv_heads, k_shape, v_shape}. For a shared layer,
                           k_shape/v_shape are the SOURCE slot's shapes (what it
                           actually reads).
    """
    n = len(geom)
    source_map = [int(g.kv_source_layer) for g in geom]
    owners = [i for i, g in enumerate(geom) if not g.is_kv_shared]

    # Pre-validate the source-map: every source must be an owner of the same
    # geometry (so the aliased slot width matches the reader's head_dim).
    for i, g in enumerate(geom):
        src = source_map[i]
        if not (0 <= src < n):
            raise ValueError(f"layer {i}: kv_source_layer {src} out of range")
        if geom[src].is_kv_shared:
            raise ValueError(f"layer {i}: source {src} is itself KV-shared")
        if geom[src].head_dim != g.head_dim or geom[src].num_kv_heads != g.num_kv_heads:
            raise ValueError(
                f"layer {i}: source {src} geometry "
                f"(hd={geom[src].head_dim}, nkv={geom[src].num_kv_heads}) != "
                f"layer (hd={g.head_dim}, nkv={g.num_kv_heads})")
        if not g.is_kv_shared and src != i:
            raise ValueError(f"non-shared layer {i} must source itself, got {src}")

    layers = []
    for i, g in enumerate(geom):
        src = source_map[i]
        # A shared layer reads its source's slot, so its effective slot shape is
        # the source's (identical width, since same type — validated above).
        k_shape, v_shape = kv_slot_shapes(
            geom[src].num_kv_heads, geom[src].head_dim, max_seq_len,
            attn_tp, batch_size, num_cores)
        layers.append({
            "idx": i,
            "owns": not g.is_kv_shared,
            "source": src,
            "head_dim": int(g.head_dim),
            "num_kv_heads": int(g.num_kv_heads),
            "k_shape": k_shape,
            "v_shape": v_shape,
        })

    return {
        "num_layers": n,
        "source_map": source_map,
        "owners": owners,
        "layers": layers,
    }
