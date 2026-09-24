"""Gemma4 (E4B text decoder) RPU adapter.

IMPORT SAFETY: the runtime registry imports this package only when Gemma4 is
selected. This module still MUST NOT touch RPU hardware at import time. The
pure-Python ``.geometry`` helpers are import-safe; all heavy work
(weight swizzle / cache / fused forward + ``graph_cache.capture``) is deferred
into ``to_rpu`` and ``Gemma4ForRPU.forward`` (call-time imports).

The fused decoder handles mixed geometry, dual head dimensions 256/512,
KV sharing, four normalization stages, QK/V normalization, dual RoPE,
PLE side inputs, layer scaling, a tied language head and softcap=30.
Sliding-window attention uses an offset KV read, with graph signatures
bucketed by ceil(total_kv/16) beyond the 512-token window.
"""
from __future__ import annotations

import threading
from numbers import Integral

from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError
from rpu_backend.runtime.registry import register_adapter

from . import geometry
from .geometry import (
    LayerGeom,
    layer_geometry,
    cpp_layer_arrays,
    geometry_dyn_dims,
    geometry_hash,
    max_head_dim,
    max_num_kv_heads,
)

HF_ARCH = "Gemma4ForConditionalGeneration"

# Default RoPE/KV-cache horizon; override with ``model._rpu_max_seq = N`` before
# ``.to('rpu')`` for longer contexts (RoPE tables + KV slots are sized to this).
DEFAULT_MAX_SEQ = 4096

_SUPPORTED_TEXT_PROFILE = (
    2560,   # hidden_size
    10240,  # intermediate_size
    42,     # num_hidden_layers
    8,      # num_attention_heads
    2,      # num_key_value_heads
    256,    # head_dim
    512,    # global_head_dim
    256,    # hidden_size_per_layer_input
    18,     # num_kv_shared_layers
    262144,  # vocab_size
    262144,  # vocab_size_per_layer_input
    131072,  # max_position_embeddings
)
_SUPPORTED_LAYER_TYPES = tuple(
    "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
    for i in range(42)
)
_BUILD_LOCK = threading.Lock()
_INSTALL_ATTRS = (
    "_rpu_gemma4_handle",
    "_rpu_gemma4_handle_finalizer",
    "_rpu_gemma4_retirement_state",
    "_rpu_gemma4_geom",
    "_rpu_gemma4_graph_cache",
    "_rpu_gemma4_ready",
    "_execution_session",
    "_rpu_last_execution_plan",
)


def _text_config(config):
    return getattr(config, "text_config", config)


def _check_profile(config) -> None:
    """Accept only the hardware-certified Gemma4-E4B text profile."""
    tc = _text_config(config)
    try:
        profile = (
            int(tc.hidden_size),
            int(tc.intermediate_size),
            int(tc.num_hidden_layers),
            int(tc.num_attention_heads),
            int(tc.num_key_value_heads),
            int(tc.head_dim),
            int(tc.global_head_dim),
            int(tc.hidden_size_per_layer_input),
            int(tc.num_kv_shared_layers),
            int(tc.vocab_size),
            int(tc.vocab_size_per_layer_input),
            int(tc.max_position_embeddings),
        )
        layer_types = tuple(tc.layer_types)
    except (AttributeError, TypeError, ValueError) as exc:
        raise UnsupportedModelError(
            "Gemma4 RPU adapter requires the complete E4B text config."
        ) from exc

    if profile != _SUPPORTED_TEXT_PROFILE or layer_types != _SUPPORTED_LAYER_TYPES:
        raise UnsupportedModelError(
            "Gemma4 RPU supports only the certified E4B text profile: "
            f"profile={_SUPPORTED_TEXT_PROFILE}, got {profile}."
        )

    unsupported_semantics = []
    if bool(getattr(tc, "attention_k_eq_v", False)):
        unsupported_semantics.append("attention_k_eq_v=True")
    raw_num_experts = getattr(tc, "num_experts", 0)
    try:
        num_experts = int(raw_num_experts or 0)
    except (TypeError, ValueError):
        unsupported_semantics.append(f"num_experts={raw_num_experts!r}")
        num_experts = 0
    if bool(getattr(tc, "enable_moe_block", False)) or num_experts:
        unsupported_semantics.append("MoE")
    if bool(getattr(tc, "use_double_wide_mlp", False)):
        unsupported_semantics.append("use_double_wide_mlp=True")
    if bool(getattr(tc, "use_bidirectional_attention", False)):
        unsupported_semantics.append("bidirectional_attention")
    if getattr(tc, "hidden_activation", None) != "gelu_pytorch_tanh":
        unsupported_semantics.append(
            f"hidden_activation={getattr(tc, 'hidden_activation', None)!r}"
        )
    if not bool(getattr(tc, "tie_word_embeddings", False)):
        unsupported_semantics.append("untied lm_head")
    if unsupported_semantics:
        raise UnsupportedModelError(
            "Gemma4 E4B config uses unsupported semantics: "
            + ", ".join(unsupported_semantics)
        )


def _validate_max_seq(
    value, *, name="model._rpu_max_seq", maximum=None
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(
            f"{name}={value} exceeds the certified model horizon {maximum}"
        )
    return int(value)


class Gemma4Adapter:
    """RPU adapter for Gemma4 E4B (text decoder MVP).

    Builds the mixed-geometry fused decoder (``torch.ops.rpu.gemma4_*``) from a
    loaded HF ``Gemma4ForConditionalGeneration`` and installs a patched
    ``forward`` that routes through it: input embed (xsqrt H) -> PLE side-input ->
    fused decoder (graph-captured) -> tied lm_head + final softcap. The HF model
    stays on CPU (embed / PLE tables / tied lm_head live there); only the decoder
    weights are swizzled onto the RPU.
    """

    arch = HF_ARCH

    def __init__(self, model, *, rpu_execution=None):
        from rpu_backend.api._execution import (
            bind_rpu_execution,
            normalize_rpu_execution,
        )
        from rpu_backend.runtime.device import (
            extract_to_device_target,
            is_rpu_device_target,
        )

        supported_execution = {"prefill": ("chunk_size",)}
        if rpu_execution is not None:
            normalize_rpu_execution(
                rpu_execution,
                entry_point="Gemma4Adapter",
                supported=supported_execution,
            )
        self.model = model
        _check_profile(model.config)
        self._rpu_execution = bind_rpu_execution(
            model,
            rpu_execution,
            entry_point="Gemma4Adapter",
            supported=supported_execution,
        )

        # Idempotent `.to('rpu')` interceptor (matches Qwen3VLAdapter): without
        # it, `from_pretrained(...).to('rpu')` hits HF's native `nn.Module.to`
        # and never reaches `to_rpu()`, so the swizzle / forward-patch is skipped
        # (the README documents this two-step path).
        if getattr(model.to, "__rpu_wrapped__", False):
            return
        original_to = model.to

        def rpu_aware_to(*args, **kwargs):
            target = extract_to_device_target(args, kwargs)
            if target is not None and is_rpu_device_target(
                target, entry_point="Gemma4Adapter.model.to"
            ):
                if len(args) > 1 or set(kwargs) - {"device"}:
                    raise ValueError(
                        "model.to('rpu') takes no extra args/kwargs; set "
                        "`model._rpu_max_seq = N` BEFORE `.to('rpu')`.")
                return self.to_rpu()
            if getattr(self.model, "_rpu_gemma4_ready", False):
                raise RuntimeError(
                    f"model.to({target!r}) rejected: the decoder is swizzled "
                    "onto the RPU (irreversible). Reload via "
                    "RPUModelForConditionalGeneration.from_pretrained(...).")
            return original_to(*args, **kwargs)

        rpu_aware_to.__rpu_wrapped__ = True
        model.to = rpu_aware_to

    @classmethod
    def preflight(cls, config) -> None:
        _check_profile(config)

    def to_rpu(self):
        model = self.model
        if getattr(model, "_rpu_gemma4_ready", False):
            resource = getattr(model, "_rpu_gemma4_retirement_state", None)
            if (resource is None or resource.handle is None or resource.failed is not None
                    or model._execution_session.stats()["state"] in {"CLOSED", "POISONED"}):
                raise RPUBackendError("Gemma4Adapter.to_rpu(): native owner is retired or poisoned")
            self._execution_session = model._execution_session
            return model
        if getattr(model, "_rpu_gemma4_install_poisoned", False):
            raise RPUBackendError(
                "Gemma4Adapter.to_rpu(): a prior install could not be fully "
                "cleaned; reload the model before retrying."
            )
        if any(hasattr(model, name) for name in _INSTALL_ATTRS):
            raise RPUBackendError(
                "Gemma4Adapter.to_rpu(): incomplete prior install state; "
                "reload the model before retrying."
            )
        _check_profile(model.config)
        max_seq = _validate_max_seq(
            getattr(model, "_rpu_max_seq", DEFAULT_MAX_SEQ),
            maximum=int(_text_config(model.config).max_position_embeddings),
        )
        if not _BUILD_LOCK.acquire(blocking=False):
            raise RPUBackendError(
                "Gemma4Adapter.to_rpu(): another Gemma4 build is in progress."
            )

        from rpu_backend.api.causal_lm import (
            _claim_live_instance,
            _release_live_instance,
        )

        claimed = False
        install_state = {"native_created": False, "cleanup_ok": True}
        try:
            if getattr(model, "_rpu_gemma4_ready", False):
                return model
            if (getattr(model, "_rpu_gemma4_install_poisoned", False)
                    or any(hasattr(model, name) for name in _INSTALL_ATTRS)):
                raise RPUBackendError(
                    "Gemma4Adapter.to_rpu(): concurrent or incomplete install state."
                )
            _claim_live_instance(model)
            claimed = True
            result = _build_rpu_model(
                model, max_seq=max_seq, _install_state=install_state
            )
            self._execution_session = model._execution_session
            return result
        except BaseException:
            if claimed:
                if install_state.get("cleanup_ok", False):
                    _release_live_instance(model)
                else:
                    model._rpu_gemma4_install_poisoned = True
            raise
        finally:
            _BUILD_LOCK.release()

    def reconfigure_rpu_execution(self, value):
        if not getattr(self.model, "_rpu_gemma4_ready", False):
            raise RuntimeError(
                "Gemma4Adapter: call to_rpu() before reconfiguring execution"
            )
        return self.model._execution_session.reconfigure(value)


def _execution_plan(model, handle, seq_len, position, horizon, graph_cache=None):
    """Shared cold/forward planner for the actual installed Gemma4 owner."""
    import torch
    from rpu_backend.runtime.decoder import plan_bounded_prefill_execution

    stage = getattr(model, "_rpu_execution", {}).get("prefill", {})
    raw_chunk = stage.get("chunk_size", "auto")
    exact_chunk = int(raw_chunk) if seq_len > 1 and raw_chunk != "auto" else None
    if position < 0 or seq_len < 1 or position + seq_len > horizon:
        raise ValueError("Gemma4 planner position+seq exceeds the RoPE/KV horizon")
    canonical_position = 0 if seq_len == 1 else position
    planning_limit = int(horizon) - int(position)
    planning_max_kv = int(position + seq_len)
    if seq_len == 1:
        # Native decode feasibility depends only on the aligned sliding mask,
        # not on its token position. Keep the real horizon check above; this
        # matches the retained Graph's windowed/16-row bucket semantics.
        text_config = getattr(model.config, "text_config", model.config)
        sliding_window = int(getattr(text_config, "sliding_window", 512))
        planning_limit = 1
        planning_max_kv = (
            min(((position + 1 + 15) // 16) * 16, int(horizon))
            if sliding_window > 0 and position + 1 > sliding_window else 1
        )
    plan_box = {}
    execution_len, _ = plan_bounded_prefill_execution(
        int(seq_len), planning_limit, 0,
        position=int(canonical_position), padding_rows=0,
        exact_chunk_size=exact_chunk,
        execution_owner=model,
        execution_stage="prefill" if seq_len > 1 else "decode",
        execution_native=("gemma4", int(handle)),
        resolve_stage_domain=lambda length: torch.ops.rpu.gemma4_resolve_stage_domain(
            handle, int(length), int(canonical_position),
            int(planning_max_kv if seq_len == 1 else position + length),
            0 if exact_chunk is None else int(exact_chunk)),
        request_id="gemma4:text:prefill_decode",
        graph_mode="RETAINED_CACHE", queue_owner_id=int(handle),
        plan_result_sink=lambda result: plan_box.__setitem__("result", result),
        physical_metadata=(("execution_generation", int(model._execution_session.generation)),),
        plan_signature=(int(planning_max_kv),),
        graph_cache=graph_cache,
    )
    if int(execution_len) != int(seq_len):
        raise RuntimeError("Gemma4 planner introduced unsupported execution padding")
    return plan_box["result"]


def _build_rpu_model(model, *, max_seq=None, _install_state=None):
    """Swizzle the decoder onto the RPU + patch ``forward`` (call-time heavy imports)."""
    import types

    import torch
    import rpu_backend
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from rpu_backend.api._execution import (
        bind_execution_session,
        execution_serialized,
    )
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource

    from .cache import Gemma4KVCache
    from .weights import make_rope_tables, set_gemma4_weights
    from . import runtime as rt

    _check_profile(model.config)
    tc = _text_config(model.config)
    geom = layer_geometry(tc)
    H = int(tc.hidden_size)
    NQ = int(tc.num_attention_heads)
    IS = int(tc.intermediate_size)
    PD = int(tc.hidden_size_per_layer_input)
    L = int(tc.num_hidden_layers)
    eps = float(tc.rms_norm_eps)
    cap = float(getattr(tc, "final_logit_softcapping", 30.0) or 30.0)
    sliding_window = int(getattr(tc, "sliding_window", 512))
    max_seq = _validate_max_seq(
        getattr(model, "_rpu_max_seq", DEFAULT_MAX_SEQ)
        if max_seq is None else max_seq,
        maximum=int(tc.max_position_embeddings),
    )
    install_state = _install_state if _install_state is not None else {}
    install_state.update(native_created=False, cleanup_ok=True)

    sd = model.state_dict()
    raw = lambda nm: sd[nm].float()

    # Decoder + PLE weights -> swizzled fp16 on RPU.
    lists, ple = rt.swizzle_decoder_and_ple(raw, geom)
    rope_tables = make_rope_tables(tc, max_seq, device="rpu", dtype=torch.float16)
    final_norm_w = raw(rt.PFX + "norm.weight").to(torch.float16).to("rpu")

    # CPU weights kept for the patched forward (input embed + PLE side + tied lm_head).
    embed_w = sd[rt.PFX + "embed_tokens.weight"]
    eptl_w = sd[rt.PFX + "embed_tokens_per_layer.weight"]
    plmp_w = sd[rt.PFX + "per_layer_model_projection.weight"]
    pln_w = sd[rt.PFX + "per_layer_projection_norm.weight"]

    gc = rpu_backend.graph.GraphCache()
    dyn = geometry_dyn_dims(geom)
    GraphSignature = rpu_backend.graph.GraphSignature

    def make_cache(max_seq_len: int | None = None):
        requested = max_seq if max_seq_len is None else _validate_max_seq(
            max_seq_len, name="max_seq_len"
        )
        if requested > max_seq:
            raise ValueError(
                f"Gemma4 cache max_seq_len={requested} exceeds the installed "
                f"RoPE horizon {max_seq}; set model._rpu_max_seq before to_rpu()."
            )
        return Gemma4KVCache(geom, max_seq_len=requested, device="rpu")

    @execution_serialized
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None,
                return_dict=None, cache_position=None, logits_to_keep=0,
                labels=None, **kwargs):
        if input_ids is None:
            raise NotImplementedError("Gemma4 RPU forward requires input_ids "
                                      "(inputs_embeds path not supported in the MVP).")
        if inputs_embeds is not None:
            raise ValueError(
                "Gemma4 RPU forward accepts input_ids only; inputs_embeds must be None."
            )
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("Gemma4 RPU forward: input_ids must be a torch.Tensor.")
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError(
                "Gemma4 RPU forward: input_ids must have shape [1, seq] with seq >= 1."
            )
        integer_dtypes = {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        }
        if input_ids.dtype not in integer_dtypes:
            raise TypeError(
                f"Gemma4 RPU forward: input_ids must use an integer dtype, got "
                f"{input_ids.dtype}."
            )
        if labels is not None:
            raise NotImplementedError(
                "Gemma4 RPU is inference-only and does not support labels/loss."
            )
        if use_cache is False:
            raise NotImplementedError(
                "Gemma4 RPU forward requires use_cache=True."
            )
        if return_dict is False:
            raise NotImplementedError(
                "Gemma4 RPU forward requires return_dict=True."
            )
        if output_attentions or output_hidden_states:
            raise NotImplementedError("Gemma4 RPU forward: output_attentions / "
                                      "output_hidden_states not supported.")
        if (isinstance(logits_to_keep, bool)
                or not isinstance(logits_to_keep, Integral)
                or logits_to_keep < 0):
            raise ValueError(
                "Gemma4 RPU forward: logits_to_keep must be a non-negative integer."
            )
        ids = input_ids.to("cpu", dtype=torch.long)
        B, S = ids.shape
        if B != 1:
            raise NotImplementedError(f"Gemma4 RPU forward batch_size={B} (only 1).")
        cache = past_key_values if past_key_values is not None else make_cache()
        if not isinstance(cache, Gemma4KVCache):
            raise TypeError(
                "Gemma4 RPU forward requires Gemma4KVCache from build_rpu_cache()."
            )
        position = int(cache.position)

        # Bounds (#3): the RoPE tables (built to `max_seq`) and the KV cache
        # (`cache.max_seq_len`) both index by absolute position; overrunning
        # either reads/writes out of range in C++ instead of failing cleanly.
        horizon = min(int(cache.max_seq_len), max_seq)
        if position + S > horizon:
            raise ValueError(
                f"Gemma4 RPU forward: position+seq ({position}+{S}) exceeds the "
                f"RoPE/KV horizon {horizon} (RoPE max_seq={max_seq}, "
                f"cache.max_seq_len={cache.max_seq_len}); rebuild with a larger "
                "`model._rpu_max_seq` before `.to('rpu')`.")

        # Consistency (#4): the fused path runs ONE causal LTM over
        # [position, position+S). Reject inputs that silently imply a different
        # attention/position layout (padding, explicit/non-contiguous positions);
        # the all-None decode path (manual loop / generate w/o padding) is a no-op.
        if attention_mask is not None:
            ones = (attention_mask.bool().all() if attention_mask.dtype == torch.bool
                    else (attention_mask == 1).all())
            if not bool(ones):
                raise NotImplementedError(
                    "Gemma4 RPU forward: only an all-ones attention_mask (no "
                    "padding) is supported; got masked positions.")
        want = torch.arange(position, position + S, dtype=torch.long)
        for _nm, _t in (("position_ids", position_ids),
                        ("cache_position", cache_position)):
            if _t is not None:
                if (not isinstance(_t, torch.Tensor)
                        or _t.dtype not in integer_dtypes):
                    raise TypeError(
                        f"Gemma4 RPU forward: {_nm} must be an integer tensor."
                    )
                got = _t.reshape(-1).to("cpu", dtype=torch.long)
                if got.numel() != S or not torch.equal(got, want):
                    raise NotImplementedError(
                        f"Gemma4 RPU forward: {_nm}={got.tolist()} disagrees with "
                        f"the cache position (expected {want.tolist()}); explicit "
                        "or non-contiguous positions are not supported.")

        hidden = (torch.nn.functional.embedding(ids, embed_w).float() * (H ** 0.5)
                  ).to(dtype=torch.float16, device="rpu")
        side = rt.compute_side_input(
            ids, embed_weight=embed_w, embed_per_layer_weight=eptl_w,
            model_projection_weight=plmp_w, projection_norm_weight=pln_w,
            num_layers=L, hidden_size=H, ple_dim=PD, eps=eps)

        # T4: when the total context exceeds the sliding window, the fused decoder
        # routes sliding(hd256) layers through a windowed MASK_2D over an offset KV
        # read. The KV-read offset + mask DMA size are constant per 16-token bucket
        # (C++ derives them from ceil(total_kv/16)), so bucket the graph signature to
        # match: <=512 stays the legacy (windowed=0) graph; >512 re-BUILDs once per
        # 16 decode tokens and REPLAYs within the bucket.
        total_kv = position + S
        windowed = 1 if total_kv > sliding_window else 0
        seq_k_bucket = (total_kv + 15) // 16 if windowed else 0
        planned = _execution_plan(self, handle, S, position, horizon, graph_cache=gc)
        selected = planned.selected
        if selected is None or not selected.stage_tuple.physical_descriptor:
            raise RuntimeError("Gemma4 planner returned no physical descriptor")
        prefill_chunk = int(selected.stage_tuple.compute_chunk)
        sig = GraphSignature(op_id="gemma4_forward", shapes=[S, H],
                             dyn_dims=dyn + [windowed, seq_k_bucket,
                                             prefill_chunk,
                                             *planned.graph_key_words()],
                             dtypes=[torch.float16])
        with gc.capture(sig):
            out = torch.ops.rpu.gemma4_forward(
                handle, hidden, cache.k_caches, cache.v_caches,
                None, position, True, 0, side,
                selected.stage_tuple.physical_descriptor)
        receipt = planned.as_dict(include_candidates=False)
        receipt.update({
            "stage": "prefill" if S > 1 else "decode",
            "component": getattr(
                self, "_rpu_execution_component_id", "language_model"
            ),
        })
        vars(self)["_rpu_last_execution_plan"] = receipt
        final_hidden = out.cpu().float().reshape(B, S, H)
        if logits_to_keep:
            final_hidden = final_hidden[:, -int(logits_to_keep):, :]
        logits = rt.softcap_lm_logits(final_hidden, embed_w, cap)   # [B, S', vocab]
        cache.position = position + S
        cache._seen_tokens = cache.position
        return CausalLMOutputWithPast(logits=logits, past_key_values=cache)

    # Configure a pending native handle only after every Python/RPU owner and
    # replacement callable is ready. Until `_rpu_gemma4_ready` is published,
    # this block owns rollback.
    tracked_attrs = (*_INSTALL_ATTRS, "forward", "build_rpu_cache")
    model_vars = vars(model)
    snapshot = {
        name: (name in model_vars, model_vars.get(name))
        for name in tracked_attrs
    }
    # Register GC authority before native create: a registration failure has
    # no native handle to leak. Keep tensors/GraphCache, never model/Session or
    # owner-bound callables, alive through both rollback and final retirement.
    resource = _InstalledNativeResource(
        model, None, torch.ops.rpu.gemma4_destroy,
        graphs=(gc,), keepalive=(lists, ple, rope_tables, final_norm_w,
                                embed_w, eptl_w, plmp_w, pln_w),
        label="Gemma4", handle_name="_rpu_gemma4_handle")
    try:
        resource.handle = handle = int(torch.ops.rpu.gemma4_create())
        install_state["native_created"] = True
        set_gemma4_weights(
            handle, geom, lists, rope_tables, final_norm_w,
            num_q_heads=NQ, num_kv_heads_max=max_num_kv_heads(geom),
            head_dim_max=max_head_dim(geom), hidden_size=H, intermediate_size=IS,
            eps=eps, qk_scale=1.0, ple=ple, ple_dim=PD,
            sliding_window=sliding_window)
        torch.ops.rpu.gemma4_set_chunk_envelope(handle, max_seq, 0)

        def invalidate_execution_graphs(_old, _new, _generation, *, force_rebuild=False):
            # clear() deliberately preserves READY. Both apply and rollback
            # retire the old layouts and permit an explicit new BUILD cycle.
            gc.begin_warmup()
            gc.clear()
            vars(model).pop("_rpu_last_execution_plan", None)

        session = bind_execution_session(
            model,
            getattr(model, "_rpu_execution", None),
            entry_point="Gemma4Adapter",
            supported={"prefill": ("chunk_size",)},
            apply=invalidate_execution_graphs,
            rollback=invalidate_execution_graphs,
            graph_mode="RETAINED_CACHE",
        )

        model._rpu_gemma4_handle = handle
        model._rpu_gemma4_retirement_state = resource
        model._rpu_gemma4_handle_finalizer = resource.finalizer
        model._rpu_gemma4_geom = geom
        model._rpu_gemma4_graph_cache = gc
        model._execution_session = session
        model.build_rpu_cache = make_cache
        model.forward = types.MethodType(forward, model)
        model._rpu_gemma4_ready = True
        install_state["committed"] = True
        return model
    except BaseException as error:
        cleanup_ok = resource.cleanup_failure(error, model, snapshot)
        if resource.handle is None and resource.finalizer.alive:
            resource.finalizer.detach()  # create failed before owning a handle

        # A publication hook may be the source of the failure and can keep
        # rejecting cleanup writes. Restore the exact instance dictionary
        # without re-entering model-defined __setattr__/__delattr__ hooks.
        model_vars = vars(model)
        for name in tracked_attrs:
            model_vars.pop(name, None)
        for name, (had_attr, value) in snapshot.items():
            if had_attr:
                model_vars[name] = value
        install_state["cleanup_ok"] = cleanup_ok
        raise


register_adapter(HF_ARCH, Gemma4Adapter)

__all__ = [
    "Gemma4Adapter",
    "HF_ARCH",
    "DEFAULT_MAX_SEQ",
    "geometry",
    "LayerGeom",
    "layer_geometry",
    "cpp_layer_arrays",
    "geometry_dyn_dims",
    "geometry_hash",
    "max_head_dim",
    "max_num_kv_heads",
]
