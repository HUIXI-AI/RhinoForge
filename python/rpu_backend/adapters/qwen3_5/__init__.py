"""Qwen3.5 adapter and multimodal orchestration entry.

Qwen3.5 (``Qwen3_5ForConditionalGeneration``) is a VL model whose text backbone
interleaves full-attention and Gated-DeltaNet (GDN) linear layers per
``text_config.layer_types``. The package follows the Hugging Face model hierarchy:

  ``Qwen3_5ForConditionalGeneration``   ← ``__init__.py``: top forward + lm_head + register
          ├── ``Qwen3_5VisionModel``      ← ``vision.py`` (experimental RPU path)
          └── ``Qwen3_5Model`` (fusion)   ← ``__init__.py``: ``_rpu_qwen3_5_forward``
                    └── ``Qwen3_5TextModel``   ← ``text.py`` (text backbone)

This module owns the top-level forward, language-model head, adapter registration,
and the public ``Qwen3_5Adapter`` export. ``text.py`` owns the text backbone and
``vision.py`` owns the controlled vision path.
"""
from __future__ import annotations

import threading
import types

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
)

from rpu_backend.api.causal_lm import _claim_live_instance
from rpu_backend.runtime.device import extract_to_device_target, is_rpu_device_target
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError
from rpu_backend.api.qwen3_5_cache import (
    QWEN3_5_OPTIONAL_PADDING_CAP,
    QWEN3_5_TOTAL_PADDING_CAP,
    Qwen3_5Cache,
)
from rpu_backend.runtime.registry import register_adapter
from rpu_backend.runtime.weights import transform_linear_weight

# Text backbone (== HF Qwen3_5TextModel). Re-export the layer classifier from
# the package entry point.
from .text import (  # noqa: F401
    _text_config,
    _validate_text_install_options,
    _validate_layer_types,
    layer_types_to_is_full,
    install_qwen3_5_text_for_rpu,
    run_qwen3_5_text,
)
# Vision tower (== HF Qwen3_5VisionModel).
from .vision import (
    install_qwen3_5_vision_for_rpu,
    fuse_visual_embeds,
)


# (hidden_size, intermediate_size, num_hidden_layers, num_key_value_heads).
# Qwen3.5-0.8B uses head_dim=256 and num_attention_heads=8.
# Dense small models only — MoE rejected.
#
# Kept at module scope so package preflight and installation share one table.
_SUPPORTED_PROFILES = frozenset({
    (1024, 3584, 24, 2),   # Qwen3.5-0.8B
    (2048, 6144, 24, 2),   # Qwen3.5-2B
    (2560, 9216, 32, 4),   # Qwen3.5-4B
    (4096, 12288, 32, 4),  # Qwen3.5-9B (untied lm_head)
})


_SWIZZLE_LOCK = threading.Lock()


_QWEN3_5_TEXT_STATE_ATTRS = (
    "handle",
    "handle_finalizer",
    "graph_cache",
    "max_seq_len",
    "num_layers",
    "hidden_size",
    "rotary_dim",
    "rope_theta",
    "mrope_section",
    "padding_budget",
    "prefill_graph",
)


def _qwen3_5_text_runtime_state(model):
    """Return the complete live text state, or ``None`` for partial state."""
    if getattr(model, "_rpu_swizzled", False) is not True:
        return None
    if getattr(model, "_rpu_swizzle_started", False) is not True:
        return None
    fusion = getattr(model, "model", None)
    if fusion is None:
        return None
    inner = getattr(fusion, "language_model", fusion)
    state = getattr(inner, "_rpu_qwen3_5", None)
    if state is None or any(
        not hasattr(state, name) for name in _QWEN3_5_TEXT_STATE_ATTRS
    ):
        return None
    finalizer = state.handle_finalizer
    if state.handle is None or finalizer is None:
        return None
    if not getattr(finalizer, "alive", False):
        return None
    if state.graph_cache is None or state.prefill_graph is None:
        return None
    if getattr(inner, "_rpu_qwen3_5_text_install_started", False) is not True:
        return None
    if getattr(model, "_lm_head_w_rpu_keepalive", None) is None:
        return None
    installed_forward = vars(model).get("forward")
    if not (
        getattr(installed_forward, "__self__", None) is model
        and getattr(installed_forward, "__func__", None)
        is _rpu_qwen3_5_forward
    ):
        return None
    return state


def _sync_adapter_text_state(adapter, state) -> None:
    adapter._handle = state.handle
    adapter._graph_cache = state.graph_cache
    adapter._rpu_is_ready = True


def _cleanup_failed_text_install(
    adapter, model, inner, had_instance_forward, original_forward
) -> None:
    """Retire a failed text install without unpoisoning mutated weights."""
    state = getattr(inner, "_rpu_qwen3_5", None) if inner is not None else None
    graph_cache = getattr(state, "graph_cache", None)
    if graph_cache is not None:
        try:
            graph_cache.clear()
        except Exception:
            pass

    finalizer = getattr(state, "handle_finalizer", None)
    if finalizer is not None and getattr(finalizer, "alive", False):
        try:
            finalizer()
        except Exception:
            pass
    if inner is not None:
        vars(inner).pop("_rpu_qwen3_5", None)
    model_state = vars(model)
    model_state.pop("_lm_head_w_rpu_keepalive", None)
    if had_instance_forward:
        model_state["forward"] = original_forward
    else:
        model_state.pop("forward", None)
    adapter._handle = None
    adapter._graph_cache = None
    adapter._rpu_is_ready = False


def _config_profile(config):
    tc = _text_config(config)
    return (int(tc.hidden_size), int(tc.intermediate_size),
            int(tc.num_hidden_layers), int(tc.num_key_value_heads))


def _check_profile(config) -> None:
    """Raise UnsupportedModelError for MoE or out-of-envelope profiles."""
    tc = _text_config(config)
    n_experts = int(getattr(tc, "num_experts", 0) or 0)
    if n_experts > 0:
        raise UnsupportedModelError(
            f"Qwen3.5 MoE (num_experts={n_experts}) is out of scope; only dense "
            "small models (0.8B/2B/4B/9B) are supported by the RPU graph-building port."
        )
    profile = _config_profile(config)
    if profile not in _SUPPORTED_PROFILES:
        raise UnsupportedModelError(
            f"Qwen3.5 profile {profile} is not supported (SPM envelope / not yet "
            f"validated). Supported profiles: {sorted(_SUPPORTED_PROFILES)}."
        )


class Qwen3_5Adapter:
    """Orchestration adapter for Qwen3.5 (== ForConditionalGeneration + fusion).

    Lifecycle:
      1. ``__init__(model)`` — profile + layer_types fail-fast guards.
      2. ``to_rpu()`` — install the text backbone C++ handle, prep lm_head,
         leave the vision tower for lazy first-image install, and replace the
         top-level forward. Returns the model.

    The adapter retains the C++ handle and per-model GraphCache.
    """

    def __init__(self, model, *, rpu_execution=None):
        from rpu_backend.api._execution import (
            bind_rpu_execution,
            normalize_rpu_execution,
        )

        supported_execution = {
            "prefill": ("chunk_size", "padding_rows", "padding_budget"),
        }
        normalized_execution = None
        if rpu_execution is not None:
            normalized_execution = normalize_rpu_execution(
                rpu_execution,
                entry_point="Qwen3_5Adapter",
                supported=supported_execution,
            )
            fixed_chunk = normalized_execution.get("prefill", {}).get(
                "chunk_size"
            )
            if isinstance(fixed_chunk, int) and fixed_chunk % 64:
                raise ValueError(
                    "Qwen3_5Adapter: prefill chunk_size must be a multiple of "
                    f"64, got {fixed_chunk}"
                )

        self.model = model
        _check_profile(model.config)
        _validate_layer_types(model.config)
        self._rpu_execution = bind_rpu_execution(
            model,
            normalized_execution if rpu_execution is not None else None,
            entry_point="Qwen3_5Adapter",
            supported=supported_execution,
        )
        self.preflight_execution(model.config, self._rpu_execution)
        state = _qwen3_5_text_runtime_state(model)
        self._handle = getattr(state, "handle", None)
        self._graph_cache = getattr(state, "graph_cache", None)
        self._rpu_is_ready = state is not None

        if getattr(model.to, "__rpu_wrapped__", False):
            return

        original_to = model.to

        def rpu_aware_to(*args, **kwargs):
            target = extract_to_device_target(args, kwargs)
            if target is not None and is_rpu_device_target(
                target, entry_point="Qwen3_5Adapter.model.to"
            ):
                if len(args) > 1 or set(kwargs) - {"device"}:
                    raise ValueError(
                        "model.to('rpu') takes no extra args/kwargs; call "
                        "Qwen3_5Adapter(model).to_rpu(max_seq_len=...) for a "
                        "non-default sequence horizon."
                    )
                return self.to_rpu()
            if (
                self._rpu_is_ready
                or getattr(self.model, "_rpu_swizzled", False)
                or getattr(self.model, "_rpu_swizzle_started", False)
            ):
                raise RPUBackendError(
                    f"model.to({target!r}) rejected: weights are swizzled for RPU "
                    "(irreversible). Reload the model to use another device."
                )
            return original_to(*args, **kwargs)

        rpu_aware_to.__rpu_wrapped__ = True
        model.to = rpu_aware_to

    @classmethod
    def preflight(cls, config) -> None:
        _check_profile(config)
        _validate_layer_types(config)

    @classmethod
    def preflight_execution(cls, config, execution_config) -> None:
        prefill = execution_config.get("prefill", {})
        padding_budget = prefill.get("padding_budget")
        if (
            isinstance(padding_budget, int)
            and padding_budget > QWEN3_5_OPTIONAL_PADDING_CAP
        ):
            raise ValueError(
                "Qwen3.5 prefill padding_budget exceeds the cache-backed "
                f"maximum {QWEN3_5_OPTIONAL_PADDING_CAP}: got "
                f"{padding_budget}. Refusing before loading model weights."
            )
        padding_rows = prefill.get("padding_rows")
        if (
            isinstance(padding_rows, int)
            and padding_rows > QWEN3_5_TOTAL_PADDING_CAP
        ):
            raise ValueError(
                "Qwen3.5 prefill padding_rows exceeds the cache-backed "
                f"maximum {QWEN3_5_TOTAL_PADDING_CAP}: got {padding_rows}. "
                "Refusing before loading model weights."
            )
        requested = prefill.get("chunk_size", "auto")
        if not isinstance(requested, int):
            return
        if requested % 64:
            raise ValueError(
                "Qwen3.5 prefill chunk_size must be a multiple of 64, got "
                f"{requested}. Refusing before loading model weights."
            )
        tc = _text_config(config)
        from .text import CERTIFIED_CHUNK_ENVELOPE
        env = CERTIFIED_CHUNK_ENVELOPE.get(
            (int(tc.num_hidden_layers), int(tc.hidden_size))
        )
        if env is not None and env[1] > 0 and requested > env[1]:
            raise UnsupportedModelError(
                f"Qwen3.5 prefill chunk_size={requested} exceeds this profile's "
                f"certified ceiling {env[1]}; max certified KV length is "
                f"{env[0]}. Refusing before loading model weights."
            )

    def to_rpu(self, max_seq_len=8192):
        """Install text backbone + lm_head and patch the top-level forward.

        Order (A→B→C→D matters): text first (creates the handle the rest read),
        lm_head after (uses ``get_output_embeddings``), lazy vision state next,
        and forward replacement last.
        """
        model = self.model
        state = _qwen3_5_text_runtime_state(model)
        if state is not None:
            _sync_adapter_text_state(self, state)
            return model
        if self._rpu_is_ready or getattr(model, "_rpu_swizzled", False):
            self._handle = None
            self._graph_cache = None
            self._rpu_is_ready = False
            raise RPUBackendError(
                "Qwen3_5Adapter.to_rpu(): the ready marker exists but text "
                "runtime ownership is incomplete; reload the model."
            )
        if getattr(model, "_rpu_swizzle_started", False):
            raise RPUBackendError(
                "Qwen3_5Adapter.to_rpu(): a prior install failed after mutation; "
                "reload the model before retrying."
            )
        # Cold handle/environment settings must fail before live ownership is
        # claimed and before dtype conversion/weight swizzling can mutate the model.
        max_seq_len, _, _ = _validate_text_install_options(max_seq_len)
        if not _SWIZZLE_LOCK.acquire(blocking=False):
            raise RPUBackendError(
                "Qwen3_5Adapter.to_rpu(): another Qwen3.5 swizzle is in progress."
            )

        inner = None
        had_instance_forward = "forward" in vars(model)
        original_forward = vars(model).get("forward")
        try:
            from rpu_backend.runtime.hw_attrs import (
                install_hw_attr_validator,
                validate_postinstall,
                validate_preinstall,
            )
            validate_preinstall(model)

            state = _qwen3_5_text_runtime_state(model)
            if state is not None:
                _sync_adapter_text_state(self, state)
                return model
            if getattr(model, "_rpu_swizzled", False):
                raise RPUBackendError(
                    "Qwen3_5Adapter.to_rpu(): the ready marker exists but text "
                    "runtime ownership is incomplete; reload the model."
                )
            if getattr(model, "_rpu_swizzle_started", False):
                raise RPUBackendError(
                    "Qwen3_5Adapter.to_rpu(): a prior install failed after "
                    "mutation; reload the model before retrying."
                )

            _claim_live_instance(model)
            model._rpu_swizzle_started = True

            # Some Qwen3.5 checkpoints keep nested weights in bf16 despite
            # from_pretrained(dtype=fp16). Cast once, but keep the lazy vision
            # tower on CPU; only the text stack moves now. The output weight is
            # copied into the swizzled keepalive below.
            model.half()
            inner = getattr(model.model, "language_model", model.model)
            _out_emb = model.get_output_embeddings()
            inner.to("rpu")

            # ── Step A: text backbone (swizzle + create + set_weights) ──
            handle = install_qwen3_5_text_for_rpu(
                inner,
                model.config,
                max_seq_len,
                execution_config=getattr(model, "_rpu_execution", None),
            )
            self._handle = handle
            # Mirror the per-model GraphCache for adapter-level diagnostics.
            self._graph_cache = inner._rpu_qwen3_5.graph_cache

            # ── Step B: lm_head (ForConditionalGeneration concern) ──
            embed = inner.embed_tokens

            # lm_head on RPU via rpu_linear (aten::linear), exactly like Qwen3 — NOT
            # a CPU fallback. tie_word_embeddings=True → lm_head weight is the tied
            # embedding matrix. embed_tokens is an nn.Embedding so it is NOT swizzled
            # by convert_linear_weights_inplace; the RPU GEMM kernel expects a
            # col-partition swizzled weight (get_linear_partition(hidden,vocab)=1).
            # Feeding the raw weight to rpu_linear would use the wrong layout.
            # Swizzle once here so both prefill and non-fused decode run on-device.
            # get_output_embeddings() returns the tied embedding when tie_word_embeddings
            # (0.8B/2B/4B) and the SEPARATE lm_head weight when NOT tied (9B). Using
            # embed.weight directly is wrong for untied models (garbage logits).
            _lm_head_w = _out_emb.weight if _out_emb is not None else embed.weight
            model._lm_head_w_rpu_keepalive = transform_linear_weight(
                _lm_head_w.detach().to("cpu").to(torch.float16).contiguous(), 1).to("rpu")

            # ── Step C: vision tower — LAZY (installed on first image forward) ──
            # Not installed here on purpose: a Qwen3.5 checkpoint is used for BOTH
            # text-only chat and vision, so eagerly swizzling the 297-tensor vision
            # tower would burden every text-only run and require the vision C++ ops
            # in the .so even when no image is ever passed. `fuse_visual_embeds`
            # (vision.py) installs it idempotently the first time pixel_values arrive,
            # leaving the text-only installation path unchanged.

            # ── Step D: replace top-level forward ──
            model.forward = types.MethodType(_rpu_qwen3_5_forward, model)
            validate_postinstall(model)
            install_hw_attr_validator(model)
            self._rpu_is_ready = True
            model._rpu_swizzled = True
            return model
        except BaseException:
            _cleanup_failed_text_install(
                self, model, inner, had_instance_forward, original_forward)
            raise
        finally:
            _SWIZZLE_LOCK.release()


def _rpu_qwen3_5_forward(self, input_ids=None, inputs_embeds=None,
                         past_key_values=None, attention_mask=None,
                         position_ids=None, pixel_values=None,
                         pixel_values_videos=None, image_grid_thw=None,
                         video_grid_thw=None, mm_token_type_ids=None,
                         logits_to_keep=0, labels=None, use_cache=None,
                         output_attentions=None, output_hidden_states=None,
                         return_dict=None, cache_position=None, **kw):
    """Top forward for Qwen3_5ForConditionalGeneration (fusion-shaped).

    embed → optional vision fuse (direct merger DMA + 3D M-RoPE) → text
    decoder (RPU) → lm_head. Fragmented image-token layouts fall back to a
    host scatter; text-only skips the vision branch entirely.
    """
    from rpu_backend.runtime.hw_attrs import mark_first_forward_done
    mark_first_forward_done(self)

    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError(
            "Qwen3.5 RPU forward requires exactly one of input_ids or "
            "inputs_embeds.")
    if not isinstance(past_key_values, Qwen3_5Cache):
        raise TypeError(
            "Qwen3.5 RPU forward requires Qwen3_5Cache.from_config(...); "
            "past_key_values=None and generic HF caches are not supported.")
    if labels is not None:
        raise NotImplementedError(
            "Qwen3.5 RPU inference does not support labels/loss computation.")
    if use_cache is False:
        raise NotImplementedError(
            "Qwen3.5 RPU inference requires use_cache=True.")
    if return_dict is False:
        raise NotImplementedError(
            "Qwen3.5 RPU inference requires return_dict=True.")
    if output_attentions:
        raise NotImplementedError(
            "Qwen3.5 RPU inference does not support output_attentions=True.")
    if output_hidden_states:
        raise NotImplementedError(
            "Qwen3.5 RPU inference does not support output_hidden_states=True; "
            "the fused output buffer is reused across forwards.")
    if pixel_values_videos is not None or video_grid_thw is not None:
        raise NotImplementedError(
            "Qwen3.5 RPU inference does not support video inputs.")
    if kw:
        raise TypeError(
            "Qwen3.5 RPU forward received unsupported keyword argument(s): "
            + ", ".join(sorted(kw)))
    if not isinstance(logits_to_keep, int) or logits_to_keep < 0:
        raise NotImplementedError(
            "Qwen3.5 RPU forward supports only a non-negative integer "
            "logits_to_keep.")

    cache = past_key_values
    if pixel_values is not None:
        if input_ids is None:
            raise ValueError(
                "Qwen3.5 image inference requires input_ids to locate image tokens.")
        if image_grid_thw is None or mm_token_type_ids is None:
            raise ValueError(
                "Qwen3.5 image inference requires image_grid_thw and "
                "mm_token_type_ids.")
        if int(cache.position) != 0:
            raise NotImplementedError(
                "Qwen3.5 image inputs are supported only at prefill position 0.")

    # The fused text path implements causal, unpadded attention. Accepting a
    # padding mask and then dropping it would silently change model semantics.
    if attention_mask is not None:
        mask_cpu = attention_mask.detach().to("cpu")
        if not bool(mask_cpu.bool().all().item()):
            raise NotImplementedError(
                "Qwen3.5 RPU forward does not support padding/zero entries in "
                "attention_mask.")
        attention_mask = None

    inner = getattr(self.model, "language_model", self.model)  # Qwen3_5TextModel

    # 1. embed (CPU→RPU prep stays OUTSIDE any capture scope).
    caller_embeds = inputs_embeds is not None
    if caller_embeds:
        hidden = inputs_embeds.to(device="rpu", dtype=torch.float16).contiguous()
    else:
        ids_rpu = (input_ids if input_ids.device.type == "rpu"
                   else input_ids.to("rpu"))
        hidden = inner.embed_tokens(ids_rpu).to(dtype=torch.float16).contiguous()

    if cache_position is not None:
        actual = cache_position.detach().to("cpu").reshape(-1).long()
        expected = torch.arange(
            int(cache.position), int(cache.position) + int(hidden.shape[1]))
        if not torch.equal(actual, expected):
            raise ValueError(
                f"Qwen3.5 cache_position={actual.tolist()} does not match "
                f"cache position {int(cache.position)} and input length "
                f"{int(hidden.shape[1])}.")

    # A new text-only conversation must not inherit the decode offset installed
    # by an earlier image prefill on the same C++ model. Decode-after-image keeps
    # it because its cache position is non-zero; reset()+refeed starts at zero.
    if pixel_values is None and cache is not None and int(cache.position) == 0:
        st = getattr(inner, "_rpu_qwen3_5", None)
        if st is not None:
            torch.ops.rpu.qwen3_5_set_mrope_position_delta(st.handle, 0)

    # 2. vision fusion; text-only skips it entirely.
    #    默认路径由 in-graph merger 直接 DMA 到 final hidden DDR；非连续
    #    image-token layout 才回退 host scatter。M-RoPE 位置计算也在本段。
    if pixel_values is not None:
        hidden, position_ids = fuse_visual_embeds(
            self, hidden, input_ids, pixel_values, image_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            past_key_values=cache)

    # 3. text decoder (RPU): returns [B, real_len, hidden].
    out = run_qwen3_5_text(
        inner, hidden, cache, attention_mask=attention_mask,
        position_ids=position_ids)

    # 4. lm_head runs on RPU with the swizzled weight. Only project the last
    # `logits_to_keep` positions; 0 keeps every prefill position for parity tests.
    if isinstance(logits_to_keep, int) and logits_to_keep > 0:
        head_in = out[:, -logits_to_keep:, :].contiguous()
    else:
        head_in = out
    logits = F.linear(head_in, self._lm_head_w_rpu_keepalive).to("cpu").to(torch.float32)
    return Qwen3_5CausalLMOutputWithPast(
        loss=None,
        logits=logits,
        past_key_values=cache,
        hidden_states=None,
        attentions=None,
        rope_deltas=getattr(self.model, "rope_deltas", None),
    )


register_adapter("Qwen3_5ForConditionalGeneration", Qwen3_5Adapter)
