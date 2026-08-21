"""Qwen3.5 text backbone (== HF ``Qwen3_5TextModel``) for RhinoForge.

This module owns config-driven layer dispatch, weight preparation, GDN mixer
setup, native-handle installation, and the per-forward RPU decoder run. Embedding
and the language-model head remain at the package orchestration level, so
``run_qwen3_5_text`` accepts embedded hidden states and returns raw decoder output.

Public surface:
  - ``layer_types_to_is_full(config)`` — per-layer full/linear dispatch, re-exported
    by the package entry point.
  - ``install_qwen3_5_text_for_rpu(text_model, config) -> handle`` — swizzle text
    weights, create the C++ model, ``set_weights``; stashes per-model forward state
    on ``text_model._rpu_qwen3_5`` (a non-``*_handle`` namespace) and returns the
    C++ handle.
  - ``run_qwen3_5_text(text_model, hidden, cache, ...) -> raw`` — drive the RPU
    decoder for one forward and return the raw output ([B, real_len, hidden]).

The package entry point owns the supported-profile guard.
"""
from __future__ import annotations

import os
import types
import weakref
from contextlib import contextmanager
from numbers import Integral

from rpu_backend.runtime import rpu_env_bool
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError
from rpu_backend.api.qwen3_5_cache import QWEN3_5_OPTIONAL_PADDING_CAP
from rpu_backend.runtime.decoder import plan_bounded_prefill_execution


def _destroy_qwen3_5_handle(handle):
    """Best-effort release for the C++ text model handle."""
    try:
        import torch
        torch.ops.rpu.qwen3_5_destroy(handle)
    except Exception:
        pass


def _validate_text_install_options(max_seq_len) -> tuple[int, int, int]:
    """Validate the cold per-handle horizon and environment knobs."""
    if (
        isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, Integral)
        or max_seq_len < 1
    ):
        raise ValueError(
            f"Qwen3.5 max_seq_len must be a positive integer, got {max_seq_len!r}"
        )
    try:
        chunk_size_cap = max(
            0, int(os.environ.get("QWEN3_5_TEXT_CHUNK", "0"))
        )
        padding_budget = int(os.environ.get(
            "QWEN3_5_TEXT_PADDING_BUDGET", "64"
        ))
    except ValueError as exc:
        raise ValueError(
            "QWEN3_5_TEXT_CHUNK and QWEN3_5_TEXT_PADDING_BUDGET must be integers"
        ) from exc
    if 0 < chunk_size_cap < 64:
        raise ValueError("QWEN3_5_TEXT_CHUNK must be 0 or at least 64")
    if padding_budget < 0:
        raise ValueError("QWEN3_5_TEXT_PADDING_BUDGET must be non-negative")
    if padding_budget > QWEN3_5_OPTIONAL_PADDING_CAP:
        raise ValueError(
            "QWEN3_5_TEXT_PADDING_BUDGET exceeds the cache-backed maximum "
            f"{QWEN3_5_OPTIONAL_PADDING_CAP}: got {padding_budget}"
        )
    return int(max_seq_len), chunk_size_cap, padding_budget


@contextmanager
def _oneshot_scope(g):
    """Record, batch, execute once, then return to PASSTHROUGH.

    Qwen3.5 prefill shapes are effectively unbounded and rarely repeat. Keeping
    each large graph would consume one cache slot and one private queue without
    creating a replay hit; decode uses GraphCache below because its shape is stable.
    """
    g.begin()
    try:
        yield
    except BaseException:
        g.abort()
        raise
    else:
        g.end()


def _text_config(config):
    """Qwen3.5 nests text fields under ``config.text_config``; unit-test
    namespaces may be flat. Return the text config (fallback: config itself)."""
    return getattr(config, "text_config", config)


def _plan_prefill_execution(real_len, physical_limit, padding_budget,
                            resolve_chunk_size, *, padding_rows="auto",
                            exact_chunk_size=None):
    """Qwen3.5 wrapper around the shared planner's mandatory 64-row grid."""
    return plan_bounded_prefill_execution(
        real_len,
        physical_limit,
        padding_budget,
        resolve_chunk_size,
        alignment=64,
        padding_rows=padding_rows,
        exact_chunk_size=exact_chunk_size,
    )


_VALID_LAYER_TYPES = {"linear_attention", "full_attention"}

# Certified chunk envelope.
#
# Keyed by (num_hidden_layers, hidden_size) -> (max_kv_len, safe chunk ceiling).
# Public exact requests are allowed only at-or-below a positive ceiling; a zero
# row certifies auto only. Deny-by-default: a size with no row here
# cannot prefill at all, because the C++ planner refuses an undeclared handle.
#
# Every profile uses 64-row GDN sub-chunks. The 128-row value is a ceiling,
# not a pin: auto remains free to select any valid value at or below it.
# Larger prefills remain deny-by-default.
CERTIFIED_CHUNK_ENVELOPE: dict[tuple[int, int], tuple[int, int]] = {
    (24, 1024): (192, 128),  # qwen3_5-0.8b
    (24, 2048): (192, 128),  # qwen3_5-2b
    (32, 2560): (192, 128),  # qwen3_5-4b
    (32, 4096): (192, 128),  # qwen3_5-9b
}


def _declare_qwen3_5_chunk_envelope(handle, num_layers, hidden_size):
    import torch as _torch
    key = (int(num_layers), int(hidden_size))
    env = CERTIFIED_CHUNK_ENVELOPE.get(key)
    if env is None:
        raise UnsupportedModelError(
            f"Qwen3.5: no certified chunk envelope for (num_hidden_layers, "
            f"hidden_size)={key}. Add an explicitly validated envelope before "
            f"running this profile."
        )
    _torch.ops.rpu.qwen3_5_set_chunk_envelope(handle, env[0], env[1])


def _validate_layer_types(config) -> None:
    """Fail-fast on a missing / wrong-length / unknown-valued layer_types."""
    tc = _text_config(config)
    lt = list(getattr(tc, "layer_types", []) or [])
    n = int(tc.num_hidden_layers)
    if len(lt) != n:
        raise UnsupportedModelError(
            f"Qwen3.5 layer_types length {len(lt)} != num_hidden_layers {n}."
        )
    bad = sorted(set(lt) - _VALID_LAYER_TYPES)
    if bad:
        raise UnsupportedModelError(
            f"Qwen3.5 unsupported layer_types {bad}; "
            f"expected only {sorted(_VALID_LAYER_TYPES)}."
        )


def layer_types_to_is_full(config):
    """Return a per-layer list[int] (1=full_attention, 0=linear_attention)."""
    _validate_layer_types(config)
    lt = list(_text_config(config).layer_types)
    return [1 if t == "full_attention" else 0 for t in lt]


def split_q_gate(qg_weight, num_heads, head_dim):
    """Split a fused Qwen3.5 q_proj weight [num_heads*head_dim*2, in] into
    (query, gate), each [num_heads*head_dim, in], matching HF's per-head
    interleaved layout (q_proj(h).view(*, num_heads, head_dim*2).chunk(2, -1)).
    Rows are [h0_query h0_gate h1_query h1_gate ...], NOT [all_query | all_gate]."""
    inn = qg_weight.shape[1]
    w = qg_weight.view(num_heads, 2 * head_dim, inn)
    query = w[:, :head_dim, :].reshape(num_heads * head_dim, inn).contiguous()
    gate = w[:, head_dim:, :].reshape(num_heads * head_dim, inn).contiguous()
    return query, gate


def _replicate_kv_heads(w, nkv, head_dim, rep):
    """GQA KV replication for FULL-ATTENTION only: [nkv*head_dim, hidden] →
    [nkv*rep*head_dim, hidden], each KV head copied `rep` times consecutively
    (repeat_interleave). With rep = NUM_CORES//nkv this makes nkv_eff == NUM_CORES so
    every core owns one complete KV head (per-core norm/rope path, no cross-core GQA
    kernel). Core i gets KV head i//rep, matching the GQA grouping. GDN is untouched —
    it keeps its own dims + runtime GQA tile; only full-attn's num_kv_heads changes."""
    if rep == 1:
        return w
    hidden = w.shape[1]
    return (w.view(nkv, head_dim, hidden)
             .repeat_interleave(rep, dim=0)
             .reshape(nkv * rep * head_dim, hidden)
             .contiguous())


def _build_partial_mrope_cos_sin(head_dim, rotary_dim, rope_theta, max_seq,
                                 dtype=None, device="rpu"):
    """Build the [max_seq, rotary_dim/2] cos/sin tables for the `partial_mrope`
    kernel. The kernel reads only rotary_dim/2 columns per token and rotates the
    first rotary_dim dims (rotate-half WITHIN rotary_dim). Indexed by token ==
    position (text-only M-RoPE degenerates to 1D partial RoPE since T==H==W);
    pos_offset selects the chunk start. inv_freq uses the rotary_dim base
    (Qwen3.5 partial_rotary_factor)."""
    import torch
    if dtype is None:
        dtype = torch.float16
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    )  # [rotary_dim/2]
    pos = torch.arange(max_seq, dtype=torch.float64)
    freqs = pos[:, None] * inv_freq[None, :]            # [max_seq, rotary_dim/2]
    return (freqs.cos().to(dtype).to(device).contiguous(),
            freqs.sin().to(dtype).to(device).contiguous())


def _apply_interleaved_mrope(freqs, mrope_section):
    """M-RoPE T/H/W interleave matching ``modeling_qwen3_5``.
    freqs: [3, 1, N, rd/2] (T/H/W). Returns [1, N, rd/2] with H/W columns strobed
    in. Text-only (T==H==W) ⇒ no-op, degenerates to arange×inv_freq."""
    import torch  # noqa: F401
    freqs_t = freqs[0].clone()
    for dim_idx, offset in enumerate((1, 2), start=1):        # H=1, W=2
        length = mrope_section[dim_idx] * 3
        freqs_t[..., slice(offset, length, 3)] = freqs[dim_idx, ..., slice(offset, length, 3)]
    return freqs_t


def _build_interleaved_mrope_cos_sin(position_ids, rotary_dim, rope_theta,
                                     mrope_section, dtype=None, device="rpu"):
    """PER-FORWARD prefill cos/sin for the `partial_mrope` kernel (B channel).
    position_ids: [3, N] int (T/H/W per TOKEN — token-indexed, NOT position-lookup).
    Returns cos/sin [N, rotary_dim/2] fp16; row t = token t's interleaved M-RoPE
    freq. Initial prefill runs at position 0, so N covers its padded token span.
    Text-only ⇒ position_ids = arange (interleave degenerates to the arange table
    === set_weights' static cos_)."""
    import torch
    if dtype is None:
        dtype = torch.float16
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    )                                                             # [rd/2]
    pos_3s = position_ids.to(torch.float64)                      # [3, N]
    inv_freq_exp = inv_freq[None, None, :, None].expand(3, 1, -1, 1)  # [3,1,rd/2,1]
    pos_exp = pos_3s[:, None, None, :]                           # [3,1,1,N]
    freqs = (inv_freq_exp @ pos_exp).transpose(2, 3)            # [3,1,N,rd/2]
    freqs_merged = _apply_interleaved_mrope(freqs, mrope_section)    # [1,N,rd/2]
    cos = freqs_merged.cos().squeeze(0).to(dtype).to(device).contiguous()
    sin = freqs_merged.sin().squeeze(0).to(dtype).to(device).contiguous()
    return cos, sin


def _get_prefill_mrope_tables(state, position_ids, rotary_dim, rope_theta,
                              mrope_section):
    """Build M-RoPE tables, or reuse an adapter-opted-in exact one-entry cache."""
    import torch

    cached = getattr(state, "prefill_rope_cache", None)
    if cached is not None and torch.equal(cached[0], position_ids):
        return cached[1], cached[2]
    cos, sin = _build_interleaved_mrope_cos_sin(
        position_ids, rotary_dim, rope_theta, mrope_section)
    if hasattr(state, "prefill_rope_cache"):
        state.prefill_rope_cache = (position_ids.clone(), cos, sin)
    return cos, sin


def _normalize_prefill_position_ids(position_ids, *, real_len, padded_len):
    """Return token-indexed T/H/W positions covering an initial prefill.

    HF generation supplies ``[4, batch, seq]`` (text + T/H/W), while direct
    forwards commonly supply only the three M-RoPE lanes.
    """
    import torch

    pid = position_ids.detach().to("cpu")
    if pid.dim() == 3:
        if pid.size(1) != 1:
            raise ValueError(
                "Qwen3.5 RPU fused prefill supports batch_size == 1 for "
                "position_ids.")
        pid = pid[:, 0, :]
    if pid.dim() != 2 or pid.size(0) not in (1, 3, 4):
        raise ValueError(
            "Qwen3.5 position_ids must be [1, seq], [3, seq], "
            "[3, 1, seq], or HF generation form [4, 1, seq].")
    if pid.size(0) == 4:
        pid = pid[1:]  # HF lane 0 is text position; lanes 1..3 are T/H/W.
    elif pid.size(0) == 1:
        pid = pid.expand(3, -1)

    real_len = int(real_len)
    padded_len = int(padded_len)
    supplied_len = int(pid.size(1))
    if supplied_len < real_len:
        raise ValueError(
            f"Qwen3.5 position_ids length {supplied_len} is shorter than "
            f"the current input length {real_len}.")

    table = pid[:, :padded_len].clone()

    # Route-B pad rows are discarded and excluded from GDN carried state, but
    # the kernel still indexes them. Continue each lane to provide in-bounds,
    # deterministic values without changing the supplied real-token positions.
    if table.size(1) < padded_len:
        count = padded_len - table.size(1)
        base = table[:, -1:] if table.size(1) else torch.zeros(
            3, 1, dtype=pid.dtype)
        steps = torch.arange(1, count + 1, dtype=pid.dtype).view(1, -1)
        table = torch.cat([table, base + steps], dim=1)
    return table.contiguous()


def _drop_raw_hf_text_weights(inner, is_full):
    """Release raw weights whose transformed copies are owned by the handle."""
    if not rpu_env_bool("RPU_QWEN3_5_FREE_HF_WEIGHTS", default=True):
        return

    def _drop(mod, name):
        sub = getattr(mod, name, None)
        w = getattr(sub, "weight", None)
        if w is not None and w.numel() > 0:
            w.data = w.data.new_empty(0)

    for i, layer in enumerate(inner.layers):
        if is_full[i]:
            for name in ("q_proj", "k_proj", "v_proj"):
                _drop(layer.self_attn, name)
        else:
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                         "out_proj", "conv1d"):
                _drop(layer.linear_attn, name)


def install_qwen3_5_text_for_rpu(
    text_model,
    config,
    max_seq_len=8192,
    *,
    execution_config=None,
    _cpu_stage_weights: bool = False,
):
    """Swizzle the text-decoder weights, create the C++ Qwen3_5 model, and
    ``set_weights``. Returns the C++ handle.

    ``text_model`` is the HF ``Qwen3_5TextModel`` backbone (``model.model`` for the
    text-only causal-LM, or ``model.model.language_model`` for the VL wrapper).
    ``config`` is the TOP model config (text fields under ``config.text_config``).

    Per-model forward state (graph cache, dims, rope params, handle finalizer,
    and the handle itself) is stashed on ``text_model._rpu_qwen3_5`` — a
    non-``*_handle`` namespace so the state is not mistaken for a separately
    installable native handle. The package-level ``to_rpu`` owns ``lm_head``.

    Full-attention layers use the Qwen3 causal-decoder path; the fused
    q_proj [query;gate] is split (query → q linear; gate stored, deferred). GDN
    layers carry their decomposed mixer weights (decode recurrent + prefill chunk).
    """
    import rpu_backend as _rb

    if hasattr(text_model, "_rpu_qwen3_5"):
        raise RPUBackendError(
            "Qwen3.5 text runtime is already installed; repeated installation "
            "would swizzle the same weights twice."
        )
    if getattr(text_model, "_rpu_qwen3_5_text_install_started", False):
        raise RPUBackendError(
            "Qwen3.5 text installation previously started an irreversible "
            "weight transform; reload the model before retrying."
        )

    max_seq_len, chunk_size_cap, padding_budget = (
        _validate_text_install_options(max_seq_len)
    )
    execution_config = {} if execution_config is None else execution_config
    prefill_config = execution_config.get("prefill", {})
    requested_chunk = prefill_config.get("chunk_size", "auto")
    exact_chunk_size = (
        0 if requested_chunk == "auto" else int(requested_chunk)
    )
    if exact_chunk_size and exact_chunk_size % 64:
        raise ValueError(
            "Qwen3.5 prefill chunk_size must be a multiple of 64, got "
            f"{exact_chunk_size}"
        )
    if exact_chunk_size:
        # Exact selection disables cap-based auto-selection.
        chunk_size_cap = 0
    padding_budget = int(
        prefill_config.get("padding_budget", padding_budget)
    )
    padding_rows = prefill_config.get("padding_rows", "auto")
    # These objects own native graph/queue resources. Prepare them before the
    # irreversible weight transform, then explicitly clear them on every failed
    # path instead of relying on Python GC.
    graph_cache = _rb.graph.GraphCache()
    try:
        prefill_graph = _rb.graph.Graph()
        return _install_qwen3_5_text_for_rpu_impl(
            text_model,
            config,
            max_seq_len=max_seq_len,
            chunk_size_cap=chunk_size_cap,
            exact_chunk_size=exact_chunk_size,
            padding_budget=padding_budget,
            padding_rows=padding_rows,
            execution_config=execution_config,
            graph_cache=graph_cache,
            prefill_graph=prefill_graph,
            cpu_stage_weights=bool(_cpu_stage_weights),
        )
    except BaseException:
        try:
            graph_cache.clear()
        except Exception:
            pass
        raise


def _install_qwen3_5_text_for_rpu_impl(
    text_model,
    config,
    *,
    max_seq_len,
    chunk_size_cap,
    exact_chunk_size,
    padding_budget,
    padding_rows,
    execution_config,
    graph_cache,
    prefill_graph,
    cpu_stage_weights,
):
    """Implementation for :func:`install_qwen3_5_text_for_rpu`.

    The public wrapper owns graph cleanup until this function publishes the
    complete ``_rpu_qwen3_5`` namespace.
    """
    import torch
    from rpu_backend.runtime.weights import (
        convert_linear_weights_inplace, transform_linear_weight,
    )

    inner = text_model
    # RPU kernels are fp16-only. Like every rpu_backend adapter, the swizzle
    # (convert_linear_weights_inplace / transform_linear_weight) PRESERVES weight
    # dtype (so int8/w8a16 quant paths stay intact) and relies on the CALLER having
    # loaded the model as fp16. Fail fast with a clear message here instead of a
    # cryptic "expected Half but found BFloat16" deep in qwen3_5_forward. Check an
    # ACTUAL weight, not config.dtype (the latter stays bfloat16 even after .half()):
    # some Qwen3.5 ckpts (e.g. 4B) carry text_config.dtype=bfloat16 which overrides
    # from_pretrained(dtype=torch.float16), so the caller must .half() the model.
    tc = _text_config(config)
    adaptive_mode = getattr(tc, "adaptive_mode", None)
    if adaptive_mode not in (None, "adaLN"):
        raise UnsupportedModelError(
            f"Qwen3.5 RPU adapter does not support adaptive_mode={adaptive_mode!r}."
        )
    is_adaptive = adaptive_mode == "adaLN"
    _wdt = (
        inner.layers[0].mlp.gate_proj.weight.dtype
        if is_adaptive
        else inner.norm.weight.dtype
    )
    if _wdt != torch.float16:
        raise UnsupportedModelError(
            f"Qwen3.5 RPU adapter needs fp16 weights (RPU kernels are fp16-only), "
            f"but the model loaded as {_wdt}. Call .half() before to_rpu(), e.g. "
            f"AutoModelForCausalLM.from_pretrained(..., dtype=torch.float16).half(). "
            f"(Some Qwen3.5 ckpts set text_config.dtype=bfloat16, which overrides "
            f"from_pretrained(dtype=fp16) — an explicit .half() is required.)"
        )
    if cpu_stage_weights and any(
        parameter.device.type != "cpu" for parameter in inner.parameters()
    ):
        raise UnsupportedModelError(
            "Qwen3.5 CPU-staged installation requires CPU FP16 source weights"
        )
    is_full = layer_types_to_is_full(config)
    if not all(is_full):
        gdn_head_dims = (
            int(tc.linear_key_head_dim),
            int(tc.linear_value_head_dim),
        )
        if gdn_head_dims != (128, 128):
            raise UnsupportedModelError(
                "Qwen3.5 GDN requires linear_key_head_dim=128 and "
                "linear_value_head_dim=128; got "
                f"{gdn_head_dims}."
            )
    if is_adaptive and not all(is_full):
        raise UnsupportedModelError(
            "Qwen3.5 adaptive_mode='adaLN' currently requires all layers to "
            "use full_attention."
        )
    head_dim = int(tc.head_dim)
    nq = int(tc.num_attention_heads)
    nkv = int(tc.num_key_value_heads)
    # Full-attention GQA (nkv < NUM_CORES): replicate each KV head so nkv_eff==NUM_CORES
    # and full-attn runs on all 8 cores, ONE whole KV head per core (per-core norm/rope
    # path, no cross-core GQA kernel). attn_tp() in C++ = min(NUM_CORES, num_kv_heads())
    # then resolves to 8 automatically. This trades KV-cache size (nkv_eff/nkv×)
    # for one complete KV head per core.
    NUM_CORES = 8
    # Replicate each full-attention KV head ``8 // nkv`` times so ``nkv_eff`` is
    # ``NUM_CORES``. GDN keeps its own dimensions and runtime GQA expansion.
    kv_rep = NUM_CORES // nkv if (nkv < NUM_CORES and NUM_CORES % nkv == 0) else 1
    nkv_eff = nkv * kv_rep

    # Swizzle every Linear except q/k/v_proj (split / replicated manually below) and
    # lm_head. lm_head is tied to embed_tokens.weight (tie_word_embeddings=True);
    # swizzling it would corrupt the shared embedding matrix. GDN linear_attn projections
    # are swizzled below for the fused mixer, so they MUST be skipped here. o_proj (not
    # skipped) row-swizzles over the default NUM_CORES cores.
    # The marker intentionally survives both success and failure. A successful
    # install is guarded by the published namespace; if an outer orchestration
    # step later fails and removes that namespace, this marker still prevents a
    # direct second install from double-swizzling the same model.
    inner._rpu_qwen3_5_text_install_started = True
    convert_linear_weights_inplace(
        inner.layers,
        skip_names={"q_proj", "k_proj", "v_proj", "lm_head",
                    "in_proj_qkv", "in_proj_z", "in_proj_b",
                    "in_proj_a", "out_proj"})

    # ── GDN mixer weight prep for the 8-core (2-head/core) fused mixer ──
    # in_proj is 8-core col-partition: each core's contiguous output slice must
    # be [2 q-heads | 2 k-heads | 2 v-heads] for heads {2c,2c+1}. So reorder
    # the conv_dim channels of in_proj_qkv (+ the depthwise conv weight + the
    # conv_state cache layout) by this permutation, THEN col-swizzle. z/b/a are
    # naturally head-ordered (col-part gives 2/core). out_proj is ROW-partition
    # (each core owns the K-slice for its 2 heads) + all_reduce.
    _gnvh = int(tc.linear_num_value_heads)
    _gnkh = int(tc.linear_num_key_heads)
    _gdk = int(tc.linear_key_head_dim)
    _gdv = int(tc.linear_value_head_dim)
    _gHc = _gnvh // NUM_CORES
    # GQA-GDN (nkh < nvh): NO weight replication — q/k stay at their true nkh heads;
    # the 16→nvh GQA expansion happens at runtime via a tile op in build_gdn.
    # in_proj_qkv = [all-q(nkh) | all-k(nkh) | all-v(nvh)].
    _gksz = _gnkh * _gdk                   # q proj size = k proj size (2048, no rep)

    def _sw_col(w):   # 8-core col-partition (no reorder: z head-ordered)
        return transform_linear_weight(
            w.detach().to("cpu").contiguous(), 1, NUM_CORES).to("rpu")

    # Per-head scalars (A_log/dt) are 2/core — too small for an aligned scatter
    # (needs >=16 B/core). Pad to 8 slots/core, core-major: core c's slots
    # [0:Hc] hold heads {2c, 2c+1}.
    def _pad_scalar(w):  # [H] → [nc*8] core-major (first Hc/core)
        out = torch.zeros(NUM_CORES * 8, dtype=w.dtype)
        wc = w.detach().to("cpu")
        for c in range(NUM_CORES):
            out[c * 8:c * 8 + _gHc] = wc[c * _gHc:(c + 1) * _gHc]
        return out.contiguous().to("rpu")

    def _sw_row(w):   # 8-core row-partition (out_proj)
        return transform_linear_weight(
            w.detach().to("cpu").contiguous(), 0, NUM_CORES).to("rpu")

    # ===== GDN per-path weights — SEPARATE q/k/v proj + per-path conv + N_bg b/a =====
    # Shared by decode (build_gdn recurrent) and prefill (chunk); matches rhino's
    # decomposed prep. in_proj_qkv = [all-q(_gksz) | all-k(_gksz) | all-v(nvh·dv)].
    _gvsz = _gnvh * _gdv                       # v proj width (4096)
    def _qkv_path(w, lo, hi):  # col-parallel proj of one path (head-ordered, no reorder)
        return transform_linear_weight(
            w.detach().to("cpu")[lo:hi].contiguous(), 1, NUM_CORES).to("rpu")
    def _conv_path(la, lo, hi):  # per-path conv slice -> [nc, Kc, (hi-lo)/nc]
        cw = la.conv1d.weight.detach().to("cpu").squeeze(1)[lo:hi]   # [hi-lo, Kc]
        Kc = cw.shape[1]
        wd = (hi - lo) // NUM_CORES
        return cw.reshape(NUM_CORES, wd, Kc).permute(0, 2, 1).contiguous().to("rpu")
    _vg_c = _gnvh // NUM_CORES                 # v heads/core (4)
    _n_bg = ((_vg_c + 15) // 16) * 16          # padded width/core (16)
    def _ba_nbg(w):  # [nvh, hidden] -> col-parallel [n_bg·nc, hidden] (first vg_c/core valid)
        hid = w.shape[1]
        out = torch.zeros(_n_bg * NUM_CORES, hid, dtype=w.dtype)
        wc = w.detach().to("cpu")
        for c in range(NUM_CORES):
            out[c * _n_bg:c * _n_bg + _vg_c] = wc[c * _vg_c:(c + 1) * _vg_c]
        return transform_linear_weight(out.contiguous(), 1, NUM_CORES).to("rpu")

    empty = torch.empty(0)
    q, k, v, o, qn, kn, ag, inrm, postrm, g, u, d = ([] for _ in range(12))
    # GDN mixer weight lists (parallel to num_layers; full slots = empty).
    g_z, g_out, g_alog, g_dt, g_nrm = ([] for _ in range(5))
    # prefill (chunk) per-path weights: separate q/k/v proj + conv, N_bg b/a
    g_q, g_k, g_v, g_cq, g_ck, g_cv, g_b_bg, g_a_bg = ([] for _ in range(8))
    unit_norm = None
    if is_adaptive:
        if cpu_stage_weights:
            unit_norm = torch.ones(
                int(tc.hidden_size), dtype=torch.float16, device="cpu"
            )
        else:
            unit_norm = inner.layers[0].mlp.gate_proj.weight.new_ones(
                int(tc.hidden_size)
            )
    for i, layer in enumerate(inner.layers):
        if not is_full[i]:
            # GDN layer: standard post_norm + MLP half collected like every
            # layer; the token mixer weights go into the GDN lists below.
            for lst in (q, k, v, o, qn, kn, ag):   # inrm filled below (unified channel)
                lst.append(empty)
            postrm.append(layer.post_attention_layernorm.weight.data + 1.0)
            g.append(layer.mlp.gate_proj.weight.data)
            u.append(layer.mlp.up_proj.weight.data)
            d.append(layer.mlp.down_proj.weight.data)
            la = layer.linear_attn
            # input_layernorm goes into the unified input_norm channel (inrm); the
            # C++ layer-level rmsnorm reads persistent "norm_w" for ALL layers.
            inrm.append(layer.input_layernorm.weight.data + 1.0)   # Qwen3_5RMSNorm 1+w
            g_z.append(_sw_col(la.in_proj_z.weight))
            g_out.append(_sw_row(la.out_proj.weight))       # row-partition
            g_alog.append(_pad_scalar(la.A_log.data))
            g_dt.append(_pad_scalar(la.dt_bias.data))
            g_nrm.append(la.norm.weight.data)   # gated norm: plain weight
            # prefill per-path (separate q/k/v proj + conv, N_bg b/a)
            _qkvw = la.in_proj_qkv.weight
            g_q.append(_qkv_path(_qkvw, 0, _gksz))
            g_k.append(_qkv_path(_qkvw, _gksz, 2 * _gksz))
            g_v.append(_qkv_path(_qkvw, 2 * _gksz, 2 * _gksz + _gvsz))
            g_cq.append(_conv_path(la, 0, _gksz))
            g_ck.append(_conv_path(la, _gksz, 2 * _gksz))
            g_cv.append(_conv_path(la, 2 * _gksz, 2 * _gksz + _gvsz))
            g_b_bg.append(_ba_nbg(la.in_proj_b.weight))
            g_a_bg.append(_ba_nbg(la.in_proj_a.weight))
            continue
        for lst in (g_z, g_out, g_alog, g_dt, g_nrm,
                    g_q, g_k, g_v, g_cq, g_ck, g_cv, g_b_bg, g_a_bg):
            lst.append(empty)
        a = layer.self_attn
        # q_proj.weight is [num_heads*head_dim*2, hidden] = per-head interleaved
        # [h0_query h0_gate h1_query h1_gate ...] (skipped by the swizzle pass →
        # still raw). Split per-head, then col-swizzle each over all NUM_CORES cores
        # (each core owns nq/NUM_CORES query heads, matching the 8-core GEMM).
        query_raw, gate_raw = split_q_gate(a.q_proj.weight.data, nq, head_dim)
        q.append(transform_linear_weight(query_raw, 1, NUM_CORES))
        ag.append(transform_linear_weight(gate_raw, 1, NUM_CORES))  # gate; deferred
        # k/v_proj skipped by the swizzle pass → raw; replicate each KV head kv_rep
        # times (nkv_eff==NUM_CORES) then col-swizzle over all 8 cores, so every core
        # owns one whole KV head (per-core norm/rope path, no cross-core GQA kernel).
        k.append(transform_linear_weight(
            _replicate_kv_heads(a.k_proj.weight.data, nkv, head_dim, kv_rep), 1, NUM_CORES))
        v.append(transform_linear_weight(
            _replicate_kv_heads(a.v_proj.weight.data, nkv, head_dim, kv_rep), 1, NUM_CORES))
        o.append(a.o_proj.weight.data)
        qn.append(a.q_norm.weight.data + 1.0); kn.append(a.k_norm.weight.data + 1.0)
        if is_adaptive:
            # G0.5 action AdaLN supplies its per-step (1+scale) tensors through
            # qwen3_5_action_forward. Identity weights still satisfy the shared
            # handle's norm-weight contract during cold installation.
            inrm.append(unit_norm)
            postrm.append(unit_norm)
        else:
            inrm.append(layer.input_layernorm.weight.data + 1.0)
            postrm.append(layer.post_attention_layernorm.weight.data + 1.0)
        g.append(layer.mlp.gate_proj.weight.data)
        u.append(layer.mlp.up_proj.weight.data)
        d.append(layer.mlp.down_proj.weight.data)

    # M-RoPE interleaved + partial rotary. The cos/sin tables contain exactly
    # rotary_dim/2 columns; the partial kernels receive rotary_dim explicitly.
    rp = getattr(tc, "rope_parameters", None) or {}
    rotary_dim = int(round(rp.get("partial_rotary_factor", 1.0) * head_dim))
    rope_theta = float(rp.get("rope_theta", getattr(tc, "rope_theta", 1e4)))
    mrope_section = [int(x) for x in rp.get("mrope_section", [])]
    # Decode cos/sin table MUST be as long as the KV cache (max_seq_len): decode
    # indexes cos_[position] with position up to max_seq_len-1, and there is NO
    # bounds check — a table shorter than the cache reads past it (garbage) once
    # decode passes the table length. Threaded from to_rpu(max_seq_len=...); the
    # 8192 remains the compatibility default for callers that do not pass a cap.
    cos, sin = _build_partial_mrope_cos_sin(
        head_dim, rotary_dim, rope_theta, max_seq=max_seq_len)

    if cpu_stage_weights:
        # G0.5 installs Vision before text. Performing swizzle or scalar norm
        # arithmetic after moving the text modules to RPU can enqueue hundreds
        # of eager kernels behind that live runtime and stall installation.
        # Keep every transformation above on CPU and upload only the finalized
        # contiguous FP16 tensors used by set_weights().
        for weights in (
            q, k, v, o, qn, kn, ag, inrm, postrm, g, u, d,
            g_z, g_out, g_alog, g_dt, g_nrm,
            g_q, g_k, g_v, g_cq, g_ck, g_cv, g_b_bg, g_a_bg,
        ):
            for index, weight in enumerate(weights):
                if weight.numel() and weight.device.type != "rpu":
                    weights[index] = weight.detach().to(
                        device="cpu", dtype=torch.float16
                    ).contiguous().to("rpu")

    gdn_nvh = int(tc.linear_num_value_heads)
    gdn_nkh = int(tc.linear_num_key_heads)
    gdn_dk = int(tc.linear_key_head_dim)
    gdn_dv = int(tc.linear_value_head_dim)
    # No GQA weight replication: conv_dim uses TRUE nkh (q/k stay nkh heads); the
    # nkh→nvh expansion is a runtime tile in build_gdn (12288 → 8192 for 4B).
    gdn_conv_dim = 2 * (gdn_nkh * gdn_dk) + gdn_nvh * gdn_dv   # 8192 for 4B
    gdn_conv_kernel = int(tc.linear_conv_kernel_dim)

    handle = torch.ops.rpu.qwen3_5_create()
    handle_finalizer = None
    try:
        handle_finalizer = weakref.finalize(
            inner, _destroy_qwen3_5_handle, handle)
        final_norm = (
            unit_norm
            if is_adaptive
            else inner.norm.weight.data + 1.0
        )
        if cpu_stage_weights and final_norm.device.type != "rpu":
            final_norm = final_norm.detach().to(
                device="cpu", dtype=torch.float16
            ).contiguous().to("rpu")
        torch.ops.rpu.qwen3_5_set_weights(
            handle, q, k, v, o, qn, kn, ag, inrm, postrm, g, u, d,
            cos, sin, final_norm, is_full,
            nq, nkv_eff, head_dim,
            int(tc.hidden_size), int(tc.intermediate_size),
            float(tc.rms_norm_eps), True, mrope_section,
            g_z, g_out, g_alog, g_dt, g_nrm,
            gdn_nvh, gdn_dk, gdn_dv, gdn_conv_dim, gdn_conv_kernel,
            g_q, g_k, g_v, g_cq, g_ck, g_cv, g_b_bg, g_a_bg)
        torch.ops.rpu.qwen3_5_set_chunk_size_cap(handle, chunk_size_cap)
        torch.ops.rpu.qwen3_5_set_prefill_chunk_size(
            handle, exact_chunk_size
        )
        # Certified chunk envelope, deny-by-default. Must precede the first
        # forward. Rows + evidence in CERTIFIED_CHUNK_ENVELOPE below.
        _declare_qwen3_5_chunk_envelope(
            handle, int(tc.num_hidden_layers), int(tc.hidden_size))
        _drop_raw_hf_text_weights(inner, is_full)

        # Per-model GraphCache keeps the stable decode signature on BUILD→REPLAY.
        # Variable-length prefill uses the bounded raw-Graph one-shot above instead;
        # uncached PASSTHROUGH is queue-safe but would rebuild every immediate op.
        #
        # Stash all per-forward state on `text_model._rpu_qwen3_5`. The namespace
        # is not itself a native handle. `run_qwen3_5_text` reads it, while the
        # adapter exposes its handle and graph cache for diagnostics.
        inner._rpu_qwen3_5 = types.SimpleNamespace(
            handle=handle,
            handle_finalizer=handle_finalizer,
            graph_cache=graph_cache,
            max_seq_len=max_seq_len,
            num_layers=len(is_full),
            hidden_size=int(tc.hidden_size),
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            padding_budget=padding_budget,
            padding_rows=padding_rows,
            execution_config=execution_config,
            exact_chunk_size=exact_chunk_size or None,
            adaptive_mode=adaptive_mode,
            prefill_plan_key=None,
            prefill_plan=None,
            # Prefill uses a dedicated Graph outside GraphCache because its
            # signature varies with real length. Each call records and executes
            # one shot, while decode retains the cache for stable replay.
            prefill_graph=prefill_graph,
        )
    except BaseException:
        try:
            graph_cache.clear()
        except Exception:
            pass
        if handle_finalizer is None:
            _destroy_qwen3_5_handle(handle)
        elif handle_finalizer.alive:
            handle_finalizer()
        raise
    return handle


def run_qwen3_5_text(text_model, hidden, cache, *, attention_mask=None,
                     position_ids=None):
    """Drive the RPU text decoder for one forward and return the RAW output.

    ``hidden`` is the already-embedded [B, seq, hidden] input (the orchestration
    top forward does the embed + any vision scatter). Returns the decoder output
    ([B, real_len, hidden]). lm_head + output wrapping happen in the orchestration
    top forward.

    ``position_ids`` (optional [3, N] token-indexed T/H/W) drives the prefill
    partial M-RoPE; None ⇒ text-only arange (identical to the static table).
    """
    import torch
    import rpu_backend as _rb

    st = text_model._rpu_qwen3_5
    handle = st.handle
    graph_cache = st.graph_cache
    num_layers = st.num_layers
    hidden_size = st.hidden_size
    rotary_dim = st.rotary_dim
    rope_theta = st.rope_theta
    mrope_section = st.mrope_section
    # CPU→RPU prep MUST stay OUTSIDE capture (a DMA recorded inside would
    # freeze the first call's pointer across replays).
    if hidden.dtype != torch.float16:
        hidden = hidden.to(torch.float16)
    if not hidden.is_contiguous():
        hidden = hidden.contiguous()
    if attention_mask is not None and attention_mask.device.type != "rpu":
        attention_mask = attention_mask.to("rpu")
    seq_len = hidden.shape[1]
    real_len = seq_len
    if cache.position + real_len > cache.max_seq_len:
        raise ValueError(
            f"Qwen3.5 cache capacity exceeded: position={cache.position}, "
            f"input={real_len}, max_seq_len={cache.max_seq_len}."
        )
    if real_len > 1 and cache.position != 0:
        raise NotImplementedError(
            "Qwen3.5 RPU multi-token prefill is supported only at cache "
            "position 0; reset the cache and re-feed the full prefix.")
    # Route B: first satisfy the mandatory 64-row GDN alignment, then optionally
    # spend the cold padding budget only when the exact C++ planner can reduce
    # the chunk count. Padding remains text-only and bounded by physical KV capacity.
    planned_chunk_size = 0
    if seq_len > 1:
        plan_key = (
            real_len,
            int(cache.allocated_max_seq_len),
            st.padding_budget,
            st.padding_rows,
            st.exact_chunk_size,
        )
        if st.prefill_plan_key == plan_key:
            pad_len, planned_chunk_size = st.prefill_plan
        else:
            pad_len, planned_chunk_size = _plan_prefill_execution(
                real_len,
                cache.allocated_max_seq_len,
                st.padding_budget,
                lambda n: torch.ops.rpu.qwen3_5_resolve_prefill_chunk_size(
                    handle, n),
                padding_rows=st.padding_rows,
                exact_chunk_size=st.exact_chunk_size,
            )
            st.prefill_plan_key = plan_key
            st.prefill_plan = (pad_len, planned_chunk_size)
        if pad_len > seq_len:
            hidden = torch.cat(
                [hidden, hidden.new_zeros(hidden.shape[0], pad_len - seq_len, hidden.shape[2])],
                dim=1)
            seq_len = pad_len
        torch.ops.rpu.qwen3_5_set_valid_prefill_len(handle, real_len)
    # PREFILL (seq_len>1) partial M-RoPE: build the per-forward interleaved cos/sin
    # (B channel) from THIS forward's 3D position_ids and stash them in the C++ model
    # (build_full_attention reads them for prefill). Token-indexed over the padded
    # initial sequence; text-only ⇒ arange. MUST stay OUTSIDE any capture scope
    # (a DMA recorded inside would freeze the first table's pointer across replays).
    if seq_len > 1 and mrope_section:
        pid = position_ids
        if pid is None:                      # text-only: T==H==W == absolute token index
            rng = torch.arange(0, cache.position + seq_len, dtype=torch.long)
            pid = rng.view(1, -1).expand(3, -1)          # [3, position+seq_len]
        else:
            pid = _normalize_prefill_position_ids(
                pid, real_len=real_len, padded_len=seq_len)
        pcos, psin = _get_prefill_mrope_tables(
            st, pid, rotary_dim, rope_theta, mrope_section)
        torch.ops.rpu.qwen3_5_set_prefill_rope(handle, pcos, psin)
    # Prefill is normally a non-retained oneshot because its lengths vary. A
    # checked-in fixed-profile adapter may provide a bounded one-entry cache.
    # Decode keeps its separate stable GraphCache entry across positions.
    is_prefill = seq_len > 1
    if is_prefill:
        prefill_cache = getattr(st, "prefill_graph_cache", None)
        if prefill_cache is None:
            # → oneshot:照录、照打包、照发,就是不留(见 _oneshot_scope)。
            ctx = _oneshot_scope(st.prefill_graph)
        else:
            # G0.5 retains one fixed-profile prefill graph. Debug BUILDs have
            # extra fixed-address SPM→DDR taps, so they must never share the
            # production cache/signature. Allocate the diagnostic cache lazily:
            # the normal path remains one cache and has no debug allocation.
            debug_export = bool(torch.rpu.get_debug_export())
            if debug_export:
                debug_cache = getattr(st, "prefill_debug_graph_cache", None)
                if debug_cache is None:
                    debug_cache = _rb.graph.GraphCache(max_entries=1)
                    st.prefill_debug_graph_cache = debug_cache
                    st.prefill_debug_graph_sig = None
                active_cache = debug_cache
                op_id = "rpu_g05_qwen3_5_prefill_debug"
                sig_attr = "prefill_debug_graph_sig"
            else:
                active_cache = prefill_cache
                op_id = "rpu_g05_qwen3_5_prefill"
                sig_attr = "prefill_graph_sig"
            sig = _rb.graph.GraphSignature(
                op_id=op_id,
                shapes=[seq_len, hidden_size],
                dyn_dims=[num_layers, real_len, planned_chunk_size],
                dtypes=[torch.float16],
            )
            old_sig = getattr(st, sig_attr, None)
            if old_sig is not None and old_sig != sig:
                active_cache.evict(old_sig)
            setattr(st, sig_attr, sig)
            ctx = active_cache.capture(sig)
    else:
        # Decode signatures omit position because mutable native parameters
        # refresh KV and M-RoPE positions on every replay. Prefill uses the
        # dedicated one-shot graph above; decode always has real_len == seq_len == 1.
        ctx = graph_cache.capture(_rb.graph.GraphSignature(
            op_id="rpu_qwen3_5",
            shapes=[seq_len, hidden_size],
            dyn_dims=[num_layers],
            dtypes=[torch.float16],
        ))
    with ctx:
        out = torch.ops.rpu.qwen3_5_forward(
            handle, hidden, cache.k_caches, cache.v_caches,
            cache.gdn_states, cache.conv_states,
            attention_mask, cache.position, True)
    if is_prefill:
        resolved = int(torch.ops.rpu.qwen3_5_get_resolved_chunk_size(handle))
        if resolved != planned_chunk_size:
            raise RuntimeError(
                "Qwen3.5 prefill dry/dispatch chunk plan drift: "
                f"planned={planned_chunk_size}, resolved={resolved}."
            )
    else:
        resolved = int(torch.ops.rpu.qwen3_5_get_resolved_chunk_size(handle))
    vars(text_model)["_rpu_last_execution_plan"] = {
        "stage": "prefill" if is_prefill else "decode",
        "logical_len": int(real_len),
        "execution_len": int(seq_len),
        "chunk_size": resolved,
        "padding_rows": int(seq_len - real_len),
        "position": int(cache.position),
    }
    # Route B: advance the cache by the REAL length (not the padded seq_len) so decode
    # continues from the real end and overwrites the padded KV; slice the padded tail rows
    # off `out` (batch=1 ⇒ the narrow view is contiguous). The GDN carried states already
    # reflect real_len (build_gdn zeroed the pad tokens via set_valid_prefill_len).
    cache.update_position(real_len)
    if real_len != seq_len:
        out = out[:, :real_len]
    return out
