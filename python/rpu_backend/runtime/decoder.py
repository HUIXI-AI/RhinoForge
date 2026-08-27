"""rpu_backend.runtime.decoder — generic decoder helpers.

Private helpers consumed directly by architecture adapters. This module
imports only from ``runtime.*`` and ``api.*``.
adapters/* depends on this module (downward); runtime/decoder.py MUST NOT import
from adapters/*.
"""
from __future__ import annotations

import os
import torch
import weakref
from typing import Type

from rpu_backend.runtime.log import _LOG


# Install native RPU dispatch while preserving the original CPU behavior.
def _install_rmsnorm_class_swap(rmsnorm_class: Type) -> None:
    """
    Patch RMSNorm class to use RPU kernel when on RPU device.

    Args:
        rmsnorm_class: The RMSNorm class (e.g., Qwen3RMSNorm, LlamaRMSNorm)

    Example:
        >>> from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        >>> _install_rmsnorm_class_swap(Qwen3RMSNorm)
    """
    def rpu_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.device.type == "rpu":
            # Use native PyTorch dispatch (torch.ops.rpu.rms_norm) instead of pybind11.
            return torch.ops.rpu.rms_norm(
                hidden_states,
                [self.weight.shape[0]],
                self.weight,
                self.variance_epsilon
            )
        else:
            # Use float32 for intermediate computation to avoid overflow
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)

    rmsnorm_class.forward = rpu_forward
    _LOG.info("Patched %s to use RPU kernel (native dispatch)", rmsnorm_class.__name__)


# Install RPU-format RoPE output while preserving the original CPU behavior.
def _install_rotary_class_swap(rotary_emb_class) -> None:
    """
    Patch RotaryEmbedding class to output kernel format cos/sin on RPU, original on CPU.

    RPU output:  [seq_len, head_dim//2] (kernel format)
    CPU output:  [batch, seq_len, head_dim] (original format, expanded)

    Args:
        rotary_emb_class: The RotaryEmbedding class (e.g., Qwen3RotaryEmbedding)
    """
    original_forward = rotary_emb_class.forward

    @torch.no_grad()
    def patched_forward(self, x, position_ids):
        if x.device.type == "rpu":
            with torch.profiler.record_function("rpu::rope_compute"):
                # Compute rotary phase in fp32 on CPU to avoid the RPU permute
                # kernel (fp16-only) that .contiguous() on the transposed fp32
                # `freqs` tensor would dispatch to. Phase precision still
                # accumulates in fp32 (matching Gemma's reference path); only
                # the final cos/sin are cast to fp16 when shipped to RPU.
                # Only the final cos/sin tensors cross to the RPU.
                inv_freq_cpu = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).cpu()
                position_ids_cpu = position_ids[:, None, :].float().cpu()

                with torch.autocast(device_type="cpu", enabled=False):
                    freqs = (inv_freq_cpu @ position_ids_cpu).transpose(1, 2)
                    emb = freqs.contiguous()
                    cos = emb.cos() * self.attention_scaling
                    sin = emb.sin() * self.attention_scaling

                return cos.to(dtype=x.dtype, device=x.device), sin.to(dtype=x.dtype, device=x.device)
        else:
            return original_forward(self, x, position_ids)

    rotary_emb_class.forward = patched_forward
    _LOG.info("Patched %s (RPU: kernel format, CPU: original)", rotary_emb_class.__name__)


def _rotary_table_capacity(rotary, model) -> int:
    """Return the full configured RoPE table length across HF versions."""
    candidates = (
        getattr(rotary, "max_position_embeddings", None),
        getattr(rotary, "max_seq_len_cached", None),
        getattr(getattr(rotary, "config", None),
                "max_position_embeddings", None),
        getattr(getattr(model, "config", None),
                "max_position_embeddings", None),
    )
    values = [int(value) for value in candidates if value is not None]
    return max(values, default=4096)


# Decoder helpers live here to avoid a runtime → adapters dependency.


def chunk_policy_key(handle: int) -> int:
    """This handle's chunk override, to be folded into its GRAPH SIGNATURE.

    The chunk plan decides the SPM layout — ctx.chunk_size feeds
    compute_params_hash_impl, so a policy change takes ensure_allocated's Path 1
    and re-derives every temp offset. A graph captured under one policy has that
    policy's offsets baked into its DMA descriptors, so two policies at one shape
    MUST be two signatures or the second forward replays the first one's graph
    under a layout it was not recorded for ("DMA bytes drift at cursor N").

    Why the override alone is enough, and why it is not the RESOLVED chunk:
      - under an explicit override, alloc_ctx.chunk_size IS the override
        in the native allocation context, so this is exactly the discriminator;
      - under auto (0), the resolved chunk is a function of the shape — already
        in the signature — and of the cap/envelope, which are frozen before the
        first forward (their setters check get_last_resolved_chunk_size() == 0).
    Calling the planner instead would be strictly more expensive for the same
    partition, and this has to be affordable per decode step, where the override
    reaches alloc_ctx.chunk_size just as it does in prefill.

    ⚠️ The "function of the shape" clause above is INCOMPLETE. See
    `prefill_position_key()` below for the half it omits.

    """
    return int(torch.ops.rpu.causal_decoder_get_chunk_size_override(handle))


def prefill_position_key(seq_len: int, position: int) -> int:
    """The resume position of a MULTI-TOKEN forward, for its graph signature.

    `chunk_policy_key` claims the resolved chunk is, under auto, a function of
    the shape. It is not: chunk validity depends on the KV CONTEXT LENGTH,
    `position + execution_len`, not only `execution_len`. One shape at two
    resume positions can therefore resolve to two chunk sizes and two SPM
    layouts. The signature keys the input to that decision rather than relying
    on a particular model profile to resolve both positions identically.

    Decode is deliberately exempt (`seq_len <= 1` -> 0) so successive steps
    share one entry. Every position-derived kernel argument is mutable and
    refreshed per replay, while the single-token SPM layout cannot change.
    Keying decode on position would create one graph per generated token.

    Prefill at position 0 also hashes to 0 and does not create a
    position-specific cache entry.
    """
    return 0 if int(seq_len) <= 1 else int(position)


def _validate_decoder_base_output(
    raw: torch.Tensor, expected_hidden_size: int
) -> None:
    """Validate the base-model output before constructing the HF result."""
    if not isinstance(raw, torch.Tensor):
        raise TypeError(
            "Decoder model base patch expected a tensor, got "
            f"{type(raw).__name__}"
        )
    shape = tuple(raw.shape)
    if raw.dim() != 3 or raw.size(-1) != expected_hidden_size:
        raise RuntimeError(
            "Decoder model base patch contract violation: expected "
            f"last_hidden_state shape [batch, seq, {expected_hidden_size}], "
            f"got {shape}. If you enabled fused lm_head via "
            "causal_decoder_set_lm_head, use _apply_fused_lm_head_for_rpu "
            "(in adapters/qwen3.py) on the Qwen3ForCausalLM instead of "
            "patching the base decoder model directly."
        )


def _destroy_causal_decoder_handle(handle: int) -> None:
    """Best-effort finalizer for a published causal-decoder handle."""
    try:
        torch.ops.rpu.causal_decoder_destroy(handle)
    except Exception:
        pass


_CAUSAL_DECODER_INSTALL_ATTRS = (
    "_rpu_deepstack_lang_layers",
    "_rpu_batch_decode_enabled",
    "_rpu_decoder_graph_cache",
    "_rpu_decoder_num_layers",
    "_rpu_decoder_hidden_size",
    "_rpu_decoder_deepstack_hash",
    "_rpu_decoder_handle",
    "_rpu_decoder_handle_finalizer",
    "_rpu_prefill_execution_alignment",
    "_rpu_execution",
    "_rpu_last_execution_plan",
)


def _causal_lm_runtime_complete(causal_lm) -> bool:
    """Return whether a top-level CausalLM owns a complete decoder install."""
    if getattr(causal_lm, "_rpu_swizzled", False) is not True:
        return False
    if getattr(causal_lm, "_rpu_swizzle_started", False) is not True:
        return False
    inner = getattr(causal_lm, "model", None)
    if inner is None:
        return False
    handle = getattr(inner, "_rpu_decoder_handle", None)
    finalizer = getattr(inner, "_rpu_decoder_handle_finalizer", None)
    graph_cache = getattr(inner, "_rpu_decoder_graph_cache", None)
    installed_forward = vars(inner).get("forward")
    forward_function = getattr(installed_forward, "__func__", None)
    return bool(
        handle is not None
        and finalizer is not None
        and getattr(finalizer, "alive", False)
        and graph_cache is not None
        and hasattr(inner, "_rpu_deepstack_lang_layers")
        and hasattr(inner, "_rpu_batch_decode_enabled")
        and getattr(inner, "_rpu_decoder_num_layers", None) is not None
        and getattr(inner, "_rpu_decoder_hidden_size", None) is not None
        and getattr(inner, "_rpu_decoder_deepstack_hash", None) is not None
        and getattr(installed_forward, "__self__", None) is inner
        and getattr(forward_function, "__name__", None)
        == "rpu_decoder_model_forward"
    )


def _cleanup_causal_decoder_install(
    inner,
    *,
    had_instance_forward: bool,
    original_instance_forward,
) -> None:
    """Best-effort cleanup for a failed irreversible outer adapter install."""
    if inner is None:
        return
    graph_cache = getattr(inner, "_rpu_decoder_graph_cache", None)
    if graph_cache is not None:
        try:
            graph_cache.clear()
        except Exception:
            pass

    finalizer = getattr(inner, "_rpu_decoder_handle_finalizer", None)
    handle = getattr(inner, "_rpu_decoder_handle", None)
    if finalizer is not None and getattr(finalizer, "alive", False):
        try:
            finalizer()
        except Exception:
            pass
    elif finalizer is None and handle is not None:
        _destroy_causal_decoder_handle(handle)

    # An outer publication failure can activate the model's custom
    # hardware-attribute guard before cleanup runs. Restore plain install metadata
    # without re-entering those hooks.
    inner_state = vars(inner)
    if had_instance_forward:
        inner_state["forward"] = original_instance_forward
    else:
        inner_state.pop("forward", None)
    for attr in _CAUSAL_DECODER_INSTALL_ATTRS:
        inner_state.pop(attr, None)


# Extra rows the plain-text prefill planner may add beyond the logical length
# to buy a better chunk plan. Read PER FORWARD (not cached in a module constant)
# so a test can compare the padded and unpadded plans in one process; 0 disables
# padding entirely. Kept modest on purpose — every padded row is real compute.
_PREFILL_PADDING_BUDGET_ENV = "RPU_CAUSAL_PREFILL_PADDING_BUDGET"
_PREFILL_PADDING_BUDGET_DEFAULT = 64


def plan_bounded_prefill_execution(
    real_len: int,
    physical_limit: int,
    padding_budget: int,
    resolve_chunk_size,
    position: int = 0,
    *,
    alignment: int = 1,
    padding_rows: str | int = "auto",
    exact_chunk_size: int | None = None,
    candidate_score=None,
) -> tuple[int, int]:
    """Choose a bounded prefill plan using the exact C++ chunk resolver.

    Returns ``(execution_len, chunk_size)``. ``alignment`` is an adapter-owned
    execution-length constraint: plain Qwen/Llama text uses 1, Qwen3-VL uses
    16, and Qwen3.5 uses 64.  Chunk sizes remain 16-aligned in C++ regardless.
    ``padding_budget`` is optional search room beyond the smallest aligned
    execution length; an integer ``padding_rows`` selects one exact plan.
    ``exact_chunk_size`` rejects the native resolver's otherwise-valid clamp
    to ``ceil16(execution_len)``.  This is what lets a caller jointly request
    an exact chunk and enough padding to make that exact chunk realizable.
    Model-specific downstream costs may be expressed with ``candidate_score``;
    it receives ``(execution_len, chunk_size, num_chunks, padding, tail_deficit)``.

    Two levels of ordering, and they are not the same question:

    * For ONE execution length the C++ query has already minimised
      ``(num_chunks, n*cs - execution_len)`` — fewest launches, then the most
      even split.
    * ACROSS execution lengths this picks
      ``(num_chunks, padding_rows, tail_deficit)``.  Padding is selected only
      when it makes a lower-launch legal plan; equal-launch plans keep the raw
      rows.  This prevents padding from hiding whether ordinary text really
      supports an off-grid logical length.

    A candidate whose final chunk would hold no real rows is rejected outright:
    that is a whole launch of pure padding.

    Shared by the text decoder and Qwen3-VL adapter so the two cannot drift;
    padding itself stays out of the C++ planner.
    """
    real_len = int(real_len)
    physical_limit = int(physical_limit)
    padding_budget = int(padding_budget)
    alignment = int(alignment)
    position = int(position)   # reporting only; the caller binds it into
                               # resolve_chunk_size, this is for the error text
    if exact_chunk_size is not None:
        if (
            isinstance(exact_chunk_size, bool)
            or not isinstance(exact_chunk_size, int)
            or exact_chunk_size <= 0
            or exact_chunk_size % 16
        ):
            raise ValueError(
                "exact_chunk_size must be a positive multiple of 16, got "
                f"{exact_chunk_size!r}"
            )
    if real_len <= 0:
        raise ValueError(f"real_len must be positive, got {real_len}")
    if padding_budget < 0:
        raise ValueError(
            f"padding_budget must be non-negative, got {padding_budget}"
        )
    if alignment < 1:
        raise ValueError(f"alignment must be positive, got {alignment}")
    if padding_rows != "auto":
        if isinstance(padding_rows, bool) or not isinstance(padding_rows, int):
            raise ValueError(
                "padding_rows must be 'auto' or a non-negative integer, got "
                f"{padding_rows!r}"
            )
        if padding_rows < 0:
            raise ValueError(
                f"padding_rows must be non-negative, got {padding_rows}"
            )

    base = ((real_len + alignment - 1) // alignment) * alignment
    if padding_rows == "auto":
        upper = min(physical_limit, base + padding_budget)
        candidates = range(base, upper + 1, alignment)
    else:
        execution_len = real_len + padding_rows
        if execution_len % alignment:
            raise ValueError(
                f"real_len={real_len} + padding_rows={padding_rows} must be "
                f"a multiple of execution alignment {alignment}"
            )
        upper = min(physical_limit, execution_len)
        candidates = (execution_len,) if execution_len <= physical_limit else ()
    best = None
    # Why each candidate was dropped. Without this the caller is told only that
    # nothing was plannable, and the planner's own reason — which distinguishes
    # "SPM budget" from "kernel-validity" from "certified envelope", three
    # different fixes — is swallowed by the `continue` below.
    rejected: list[str] = []
    for execution_len in candidates:
        try:
            chunk_size = int(resolve_chunk_size(execution_len))
        except RuntimeError as e:
            rejected.append(f"{execution_len}: {e}")
            continue
        if chunk_size <= 0:
            raise RuntimeError(
                "causal prefill planner returned non-positive "
                f"chunk_size={chunk_size} for execution_len={execution_len}"
            )
        if chunk_size % 16:
            raise RuntimeError(
                "causal prefill planner returned non-16-aligned "
                f"chunk_size={chunk_size} for execution_len={execution_len}"
            )
        if exact_chunk_size is not None and chunk_size != exact_chunk_size:
            rejected.append(
                f"{execution_len}: native resolver returned chunk_size="
                f"{chunk_size}, not exact request {exact_chunk_size}"
            )
            continue
        num_chunks = (execution_len + chunk_size - 1) // chunk_size
        # Do not emit a final chunk containing only padded rows.
        if (num_chunks - 1) * chunk_size >= real_len:
            rejected.append(
                f"{execution_len}: final chunk would contain only padding"
            )
            continue
        padding = execution_len - real_len
        tail_deficit = num_chunks * chunk_size - execution_len
        score = (
            candidate_score(
                execution_len,
                chunk_size,
                num_chunks,
                padding,
                tail_deficit,
            )
            if candidate_score is not None
            else (num_chunks, padding, tail_deficit)
        )
        candidate = (score, execution_len, chunk_size)
        if best is None or score < best[0]:
            best = candidate

    if best is None:
        hint = ""
        if position % 16:
            # The binding constraint is the start position, not the padded
            # execution length, so searching longer lengths cannot help.
            hint = (f" — position={position} is not 16-aligned, and a "
                    f"MULTI-TOKEN forward must start on the 16 grid (a "
                    f"1-token decode step need not). A continuation is only "
                    f"plannable after a turn whose length was a multiple of "
                    f"16; searching longer execution lengths will not fix it.")
        why = ""
        if rejected:
            # One line per candidate length: they usually share a cause, and
            # when they do not, the difference IS the diagnosis.
            why = "\n  planner rejected each candidate:\n    " + "\n    ".join(
                rejected)
        elif not candidates:
            why = (f"\n  no candidate was even tried: physical_limit="
                   f"{physical_limit} is below the requested/aligned "
                   f"execution length {base}.")
        else:
            why = ("\n  every candidate planned, but each would end in a chunk "
                   "of pure padding.")
        raise RuntimeError(
            "causal prefill has no safe execution plan for "
            f"real_len={real_len}, physical_limit={physical_limit}, "
            f"padding_budget={padding_budget}, padding_rows={padding_rows!r}, "
            f"alignment={alignment}{hint}{why}"
        )
    return best[1], best[2]


def _text_prefill_execution_plan(
    model, handle, past_key_values, seq_len: int
) -> tuple[int, int]:
    """Execution length and chunk for a plain 1D-RoPE text prefill.

    Applies to any multi-token forward of a plain text decoder — the initial
    prefill AND a continuation at position > 0 (a second conversational turn) —
    with no caller-supplied mask and no M-RoPE position_ids (Qwen3-VL plans its
    own padding upstream, including the visual embeds and the 3D position ids,
    and must not be padded twice).

    Padded rows are provably inert for the logical prefix, at any position:
    under the causal lower-triangular mask row i attends only to columns <= i,
    and every padded row sits after every real row OF THIS FORWARD. Their KV
    entries land beyond ``cache.position`` (advanced by the LOGICAL length), so
    decode overwrites the first of them and SDPA never reads the rest.
    ``physical_limit`` below is the room LEFT in the cache, not its capacity, so
    a continuation can never plan past the end.
    """
    stage = getattr(model, "_rpu_execution", {}).get("prefill", {})
    padding_rows = stage.get("padding_rows", "auto")
    if isinstance(padding_rows, int) and not isinstance(padding_rows, bool):
        budget = 0  # exact rows do not consult the optional-search budget
    else:
        try:
            budget = int(stage.get(
                "padding_budget",
                os.environ.get(
                    _PREFILL_PADDING_BUDGET_ENV,
                    _PREFILL_PADDING_BUDGET_DEFAULT,
                ),
            ))
        except ValueError as exc:
            raise ValueError(
                f"{_PREFILL_PADDING_BUDGET_ENV} must be an integer") from exc
        if budget < 0:
            raise ValueError(
                f"{_PREFILL_PADDING_BUDGET_ENV} must be non-negative"
            )
    physical_limit = int(past_key_values.max_seq_len) - int(past_key_values.position)
    execution_len, chunk_size = plan_bounded_prefill_execution(
        seq_len, physical_limit, budget,
        # The resolver takes the POSITION this forward will start at: chunk
        # validity depends on the KV context length, so planning a
        # continuation at 0 accepts a split compute_chunks then rejects.
        lambda n: torch.ops.rpu.causal_decoder_resolve_prefill_chunk_size(
            handle, n, int(past_key_values.position)),
        position=int(past_key_values.position),
        alignment=1,
        padding_rows=padding_rows,
        exact_chunk_size=(
            stage.get("chunk_size")
            if isinstance(stage.get("chunk_size"), int)
            else None
        ),
    )
    return int(execution_len), int(chunk_size)


def _canonicalize_plain_text_controls(
    seq_len: int,
    past_key_values,
    attention_mask,
    position_ids,
    cache_position,
    *,
    batch_size: int = 1,
):
    """Validate no-op HF controls, then remove them for the fused text path.

    ``generate()`` normally supplies an all-ones mask and contiguous positions.
    Those inputs describe the same unpadded causal execution that the fused
    kernel implements, so keeping them non-``None`` must not bypass bounded
    prefill planning.  Anything non-trivial is unsupported and must fail before
    graph capture rather than being silently ignored by the causal-only kernel.
    """
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError(
                "RPU plain-text attention_mask must be a tensor or None"
            )
        mask_cpu = attention_mask.detach().to("cpu")
        all_ones = (
            mask_cpu.bool().all()
            if mask_cpu.dtype == torch.bool
            else (mask_cpu == 1).all()
        )
        if mask_cpu.numel() == 0 or not bool(all_ones.item()):
            raise NotImplementedError(
                "RPU plain-text fused forward supports only an all-ones "
                "attention_mask; padding/zero entries are not supported"
            )
        expected_mask_shape = (
            int(batch_size),
            int(past_key_values.position) + int(seq_len),
        )
        if tuple(mask_cpu.shape) != expected_mask_shape:
            raise NotImplementedError(
                "RPU plain-text fused forward requires attention_mask shape "
                f"{expected_mask_shape} for the current cache position; got "
                f"{tuple(mask_cpu.shape)}"
            )

    expected = torch.arange(
        int(past_key_values.position),
        int(past_key_values.position) + int(seq_len),
        dtype=torch.long,
    )
    integer_dtypes = (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    for name, value in (
        ("position_ids", position_ids),
        ("cache_position", cache_position),
    ):
        if value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.dtype not in integer_dtypes:
            raise TypeError(
                f"RPU plain-text {name} must be an integer tensor or None"
            )
        actual = value.detach().reshape(-1).to("cpu", dtype=torch.long)
        expected_values = expected
        if int(batch_size) > 1 and actual.numel() == int(batch_size) * seq_len:
            expected_values = expected.repeat(int(batch_size))
        if not torch.equal(actual, expected_values):
            raise NotImplementedError(
                f"RPU plain-text fused forward requires {name} to match "
                f"the contiguous cache range [{int(past_key_values.position)}, "
                f"{int(past_key_values.position) + int(seq_len)}); got "
                f"{actual.tolist()}"
            )

    return None, None, None


def _reject_unconsumed_text_padding(
    model,
    seq_len: int,
    attention_mask,
    position_ids,
) -> None:
    """Reject an explicit padding policy when this path cannot apply it."""
    stage = getattr(model, "_rpu_execution", {}).get("prefill", {})
    padding_rows = stage.get("padding_rows")
    padding_budget = stage.get("padding_budget", 0)
    requests_padding = (
        padding_rows == "auto"
        or (isinstance(padding_rows, int) and padding_rows > 0)
        or (isinstance(padding_budget, int) and padding_budget > 0)
    )
    if (
        seq_len > 1
        and requests_padding
        and (attention_mask is not None or position_ids is not None)
    ):
        raise ValueError(
            "rpu_execution prefill padding cannot be applied "
            "when attention_mask or position_ids is supplied; pass "
            "padding_rows=0 with no padding_budget, or omit the explicit "
            "mask/positions"
        )


_BATCH_PREFILL_ALL_ONES_MASK = object()


# Shared causal-decoder forward runner.
def _run_causal_decoder_forward(
    model,
    handle,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    deepstack_dense_visual_embeds=None,
    rope_cos_il=None,
    rope_sin_il=None,
    prefill_plan=None,
    batch_slot=0,
    _validated_all_ones_attention_mask=None,
    **kwargs,
):
    """Shared RPU forward runner for the all-layers-once causal-decoder kernel.

    This body is cross-architecture and uses the generic
    ``torch.ops.rpu.causal_decoder_forward`` operation family.

    Returns:
        (raw_tensor, past_key_values)
        raw_tensor is [batch, seq_len, hidden_size] hidden states.
        The native operation returns hidden states; outer wrappers own any
        shape-changing logits projection.
    """
    # Late import (avoids module-load circular with api.cache).
    from rpu_backend.api.cache import RPUCache
    # The first-forward marker is idempotent.
    from rpu_backend.runtime.hw_attrs import mark_first_forward_done
    mark_first_forward_done(model)

    # HF contract: input_ids XOR inputs_embeds
    if (input_ids is None) == (inputs_embeds is None):
        raise AssertionError(
            "RPU all-layers-once: must provide exactly one of "
            "input_ids or inputs_embeds"
        )

    # past_key_values must be an RPUCache instance
    if not isinstance(past_key_values, RPUCache):
        raise AssertionError(
            f"RPU all-layers-once: past_key_values must be an "
            f"RPUCache instance, got {type(past_key_values).__name__}. "
            f"Create one via: "
            f"cache = rpu_backend.RPUCache(num_layers, batch_size, max_seq_len, "
            f"num_kv_heads, head_dim)"
        )

    # Unsupported HF options — fail-fast instead of silent drop
    if output_attentions:
        raise AssertionError(
            "RPU all-layers-once: output_attentions is not supported"
        )
    if output_hidden_states:
        raise AssertionError(
            "RPU all-layers-once: output_hidden_states is not supported"
        )
    if return_dict is False:
        raise AssertionError(
            "RPU all-layers-once: return_dict=False is not supported "
            "because this path returns OutputWithPast"
        )
    if use_cache is False:
        raise AssertionError(
            "RPU all-layers-once: use_cache=False is not supported "
            "because this is a cache-backed inference path"
        )
    # cache_position is received but ignored (RPUCache.position owns position).
    _ = cache_position
    # Forward position_ids to the native operation when Qwen3-VL M-RoPE is active.
    # M-RoPE is active. Non-mrope models (Qwen3 / Llama) pass position_ids=None
    # to the op — the C++ side silently ignores it.
    #
    # HF's Qwen3-VL forward emits position_ids in shape [3, batch, seq_len];
    # the C++ M-RoPE entry check expects [seq_len, 3] int32 RPU contig.
    # Normalize here so adapters don't each duplicate the reshape.
    if position_ids is not None and position_ids.dim() == 3:
        # [3, batch, seq_len] → [seq_len, 3]; only batch==1 is supported.
        if position_ids.size(0) != 3 or position_ids.size(1) != 1:
            raise AssertionError(
                f"RPU all-layers-once: 3D position_ids must be [3, batch=1, seq_len], "
                f"got {tuple(position_ids.shape)}"
            )
        position_ids = position_ids.squeeze(1).transpose(0, 1).contiguous()
    if position_ids is not None:
        if position_ids.dtype != torch.int32:
            position_ids = position_ids.to(torch.int32)
        if position_ids.device.type != "rpu":
            position_ids = position_ids.to("rpu")
        if not position_ids.is_contiguous():
            position_ids = position_ids.contiguous()

    # Input embedding
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = model.embed_tokens(input_ids)

    batch_size = int(hidden_states.shape[0])
    seq_len = int(hidden_states.shape[1])
    batch_decode_enabled = bool(
        getattr(model, "_rpu_batch_decode_enabled", False)
    )

    # Batch decode: B sequences pack into the GEMM M dim so a decode step reads
    # each weight once for all of them. Only seq_len == 1 qualifies — at
    # seq_len > 1 the rows of one sequence carry consecutive RoPE positions, so
    # a packed [B, S] block would need per-row positions and a block-diagonal
    # causal mask. Batched prefill is therefore driven as one call per sequence.
    # The C++ side TORCH_CHECKs the same pair.
    if batch_size > 1 and not batch_decode_enabled:
        raise AssertionError(
            "RPU all-layers-once: batch > 1 is enabled only for the Qwen3 "
            "decoder capability"
        )
    if batch_size > 1 and seq_len != 1:
        raise AssertionError(
            f"RPU all-layers-once: batch > 1 requires seq_len == 1 (decode); got "
            f"batch={batch_size}, seq_len={seq_len}. "
            f"Run batched prefill one sequence at a time."
        )
    if batch_size > 1 and past_key_values.batch_size != batch_size:
        raise AssertionError(
            "RPU all-layers-once: batched decode requires one KV slot per "
            f"sequence; hidden batch={batch_size}, cache batch_size="
            f"{past_key_values.batch_size}"
        )
    if not 0 <= int(batch_slot) < int(past_key_values.batch_size):
        raise AssertionError(
            "RPU all-layers-once: batch_slot must index the RPUCache; got "
            f"batch_slot={batch_slot}, cache batch_size="
            f"{past_key_values.batch_size}"
        )

    # Must be half + contiguous + RPU device for the fused kernel
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if hidden_states.dtype != torch.float16:
        hidden_states = hidden_states.to(torch.float16)

    # Bounded prefill planning (text decoder). Plain text starts from the raw
    # logical length; optional padding is admitted only when the exact resolver
    # finds a lower-launch legal plan. Deliberately narrow:
    # 1D RoPE only (M-RoPE callers pad upstream) and either no caller mask or
    # the all-ones mask already validated by `_run_batched_prefill`. Keep that
    # logical mask intact; causal fused SDPA does not consume an explicit mask.
    #
    # NOT restricted to position == 0. A continuation — a second conversational
    # turn, i.e. a multi-token forward on top of the KV an earlier turn wrote —
    # needs the same safe-tail planning as the first turn, and every
    # step of the safety argument is position-independent: the padded rows are
    # appended AFTER all real rows OF THIS FORWARD, the causal mask stops real
    # rows from attending to them, their KV lands at
    # [position+seq_len, position+execution_len), and update_position() below
    # advances by the LOGICAL length — so decode overwrites the first padded row
    # and SDPA never reads the rest. plan_bounded_prefill_execution already
    # takes the REMAINING physical room (max_seq_len - position) as its limit.
    #
    execution_len = seq_len
    planned_chunk_size = 0
    if (seq_len > 1
            and position_ids is None
            and (attention_mask is None
                 or _validated_all_ones_attention_mask
                 is _BATCH_PREFILL_ALL_ONES_MASK)):
        if prefill_plan is None:
            prefill_plan = _text_prefill_execution_plan(
                model, handle, past_key_values, seq_len)
        execution_len, planned_chunk_size = map(int, prefill_plan)
        if execution_len < seq_len:
            raise RuntimeError(
                f"prefill execution_len={execution_len} is smaller than "
                f"logical seq_len={seq_len}"
            )
        if execution_len > seq_len:
            hidden_states = torch.cat(
                [hidden_states,
                 hidden_states.new_zeros(hidden_states.shape[0],
                                         execution_len - seq_len,
                                         hidden_states.shape[2])],
                dim=1,
            ).contiguous()

    # Forward dense visual embeddings when Qwen3-VL DeepStack is active.
    # C++ op when DeepStack is active. None default → empty list (transparent
    # for Qwen3 / Llama 1D-RoPE callers). Qwen3-VL callers pass either
    # zero-keepalive views (text-only / decode) or scattered dense embeds
    # (image prefill).
    if deepstack_dense_visual_embeds is None:
        deepstack_dense_visual_embeds = []

    # Call the fused all-layers-once C++ op.
    # is_causal=True is HARDCODED here. See comment block above.
    raw = torch.ops.rpu.causal_decoder_forward(
        handle,
        hidden_states,
        past_key_values.k_caches,
        past_key_values.v_caches,
        attention_mask,
        past_key_values.position,
        True,  # is_causal — fused SDPA is causal-only
        position_ids,                   # forwarded to M-RoPE; None otherwise
        deepstack_dense_visual_embeds,  # DeepStack injection or []
        rope_cos_il,                    # partial_mrope: host-baked interleaved cos (None → legacy
        rope_sin_il,                    #   in-kernel llama_mrope_interleave; both None = unchanged)
        -1,                             # cos_sin_offset: -1 = legacy ctx().position base
        batch_slot,                     # batch decode: KV-cache slot for this batch==1 call
        batch_decode_enabled,           # explicit per-model capability; Qwen3 only
    )

    if planned_chunk_size:
        resolved_chunk_size = int(
            torch.ops.rpu.causal_decoder_get_resolved_chunk_size(handle)
        )
        if resolved_chunk_size != planned_chunk_size:
            raise RuntimeError(
                "causal prefill dry/forward chunk mismatch: "
                f"planned={planned_chunk_size}, resolved={resolved_chunk_size}, "
                f"logical_len={seq_len}, execution_len={execution_len}"
            )
    else:
        resolved_chunk_size = int(
            torch.ops.rpu.causal_decoder_get_resolved_chunk_size(handle)
        )

    vars(model)["_rpu_last_execution_plan"] = {
        "stage": "prefill" if seq_len > 1 else "decode",
        "logical_len": int(seq_len),
        "execution_len": int(execution_len),
        "chunk_size": resolved_chunk_size,
        "padding_rows": int(execution_len - seq_len),
        "position": int(past_key_values.position),
    }

    # Slice the padded tail off before anyone sees it, and advance the cache by
    # the LOGICAL length so decode continues from the real end (and overwrites
    # the first padded KV row).
    if execution_len > seq_len:
        raw = raw[:, :seq_len]

    # Global position advance (the fused kernel does not touch RPUCache state)
    past_key_values.update_position(seq_len)

    return raw, past_key_values


def _is_batched_prefill(input_ids, inputs_embeds) -> bool:
    """True when the request is [B>1, S>1] — the shape the fused kernel cannot
    take in one call (see the batch gate in `_run_causal_decoder_forward`)."""
    src = inputs_embeds if inputs_embeds is not None else input_ids
    if src is None or getattr(src, "dim", None) is None or src.dim() < 2:
        return False
    return int(src.shape[0]) > 1 and int(src.shape[1]) > 1


def _run_batched_prefill(model, handle, capture_factory, *, past_key_values,
                         input_ids=None, inputs_embeds=None,
                         prefill_plan=None, **kwargs):
    """Prefill a [B, S] batch as B captured single-sequence forwards.

    Each sequence writes its own KV-cache slot via `batch_slot=b`. The cache
    base lives in mutable kernel registers that `sync_mutable_params()`
    re-pushes on every REPLAY (exactly like `position`), so all B calls share
    ONE graph: sequence 0 BUILDs it, sequences 1..B-1 REPLAY it. That is why
    `batch_slot` must NOT appear in the GraphCache signature.

    Every row must contain the same number of real tokens. ``attention_mask``
    may be omitted as an explicit caller assertion that this is true, or it may
    be an all-ones ``[B, S]`` tensor. Padding masks are rejected because all
    rows share one logical decode position. Contiguous ``position_ids`` and
    ``cache_position`` supplied by ``generate()`` are validated and removed.
    The position is rewound before each row and left advanced once at the end.

    `capture_factory(batch, seq_len, prefill_plan)` returns the capture context
    for a single sequence — the caller owns signature construction.
    """
    src = inputs_embeds if inputs_embeds is not None else input_ids
    batch, seq_len = int(src.shape[0]), int(src.shape[1])
    if not bool(getattr(model, "_rpu_batch_decode_enabled", False)):
        raise AssertionError(
            "RPU batched prefill is enabled only for the Qwen3 decoder "
            "capability"
        )
    if int(past_key_values.batch_size) != batch:
        raise AssertionError(
            "RPU batched prefill requires one KV slot per sequence; "
            f"input batch={batch}, cache batch_size="
            f"{past_key_values.batch_size}"
        )

    attention_mask = kwargs.pop("attention_mask", None)
    position_ids = kwargs.pop("position_ids", None)
    cache_position = kwargs.pop("cache_position", None)
    if attention_mask is not None:
        expected_mask_shape = (
            batch,
            int(past_key_values.position) + seq_len,
        )
        if not isinstance(attention_mask, torch.Tensor) or tuple(
            attention_mask.shape
        ) != expected_mask_shape:
            raise AssertionError(
                "RPU batched prefill attention_mask must have shape "
                f"{list(expected_mask_shape)} and contain only ones"
            )
        mask_check = attention_mask.detach()
        if mask_check.device.type != "cpu":
            mask_check = mask_check.to("cpu")
        if not bool(torch.all(mask_check == 1).item()):
            raise AssertionError(
                "RPU batched prefill does not support padding: "
                "attention_mask must contain only ones"
            )

    has_explicit_positions = (
        position_ids is not None or cache_position is not None
    )
    if has_explicit_positions:
        for b in range(batch):
            position_ids_b = position_ids
            if (
                isinstance(position_ids, torch.Tensor)
                and position_ids.dim() == 2
                and int(position_ids.shape[0]) == batch
            ):
                position_ids_b = position_ids[b:b + 1]
            _canonicalize_plain_text_controls(
                seq_len,
                past_key_values,
                None,
                position_ids_b,
                cache_position,
            )

    # The outer wrappers normally plan before entering this helper. HF
    # generate() supplies explicit positions, so canonicalize those no-op
    # controls here and establish the physical plan before graph capture.
    if prefill_plan is None and has_explicit_positions:
        prefill_plan = _text_prefill_execution_plan(
            model, handle, past_key_values, seq_len
        )
    base_position = past_key_values.position

    try:
        outs = []
        for b in range(batch):
            past_key_values.reset_to_position(base_position)
            ids_b = None if input_ids is None else input_ids[b:b + 1]
            emb_b = None if inputs_embeds is None else inputs_embeds[b:b + 1]
            mask_b = (
                None if attention_mask is None else attention_mask[b:b + 1]
            )
            with capture_factory(1, seq_len, prefill_plan):
                raw_b, _ = _run_causal_decoder_forward(
                    model, handle,
                    input_ids=ids_b,
                    inputs_embeds=emb_b,
                    attention_mask=mask_b,
                    past_key_values=past_key_values,
                    batch_slot=b,
                    _validated_all_ones_attention_mask=(
                        _BATCH_PREFILL_ALL_ONES_MASK
                        if mask_b is not None else None
                    ),
                    prefill_plan=prefill_plan,
                    **kwargs,
                )
            # The C++ output tensor is a per-SHAPE registry slot, so every b
            # returns the SAME tensor. Copy before the next call overwrites it.
            outs.append(raw_b.clone())

        return torch.cat(outs, dim=0), past_key_values
    except BaseException:
        past_key_values.reset_to_position(base_position)
        raise


# Generic CausalLM all-layers-once fusion seam shared by Qwen3 and Llama.
# Qwen3-specific q_norm/k_norm reads stay on the Qwen3 surface, behind the
# `qk_norm_lists=` kwarg gate.
#
# All callers read ``_rpu_decoder_handle`` directly.
def _install_causal_decoder_forward(
    model,
    *,
    arch: str,
    qk_norm_lists=None,
    mrope_section=None,
    cos_sin=None,
    deepstack_lang_layers=None,
    scale_lists=None,
    chunk_envelope_for,
    execution_config=None,
) -> int:
    """Generic CausalLM all-layers-once installer shared by Qwen3 and Llama.

    The C++ kernel side accepts empty per-layer q_norm/k_norm tensor lists and routes on
    an explicit `has_qk_norm_` boolean. Qwen3 callers pass the gathered q/k norm
    lists; Llama callers pass `qk_norm_lists=None` (which materializes as empty
    lists at the C++ boundary).

    **Transactional + idempotent**: a replacement handle is fully configured
    before the old published handle is retired. Any prepare/create/set-weights
    failure destroys only the new handle and leaves the old model state intact.

    Prerequisites:
      - model.to("rpu") must have been called (weights on RPU device)
      - convert_linear_weights_inplace(model) must have been called
        (applies row/col partition swizzle to Linear weights)

    Args:
        model: A decoder model instance (Qwen3Model or LlamaModel — the inner base, not
            the outer ForCausalLM wrapper).
        arch: "qwen3" — Qwen3-specific q_norm/k_norm gathered from
              model.layers[i].self_attn.q_norm/k_norm.
              "llama" — empty q_norm/k_norm tensor lists passed to the native
              operation (has_qk_norm_=false branch with in-place RoPE).
              "qwen3_vl_text" — Qwen3-VL text decoder; same QK-norm layout as Qwen3,
              additionally requires `mrope_section` and kernel-format `cos_sin`.
        qk_norm_lists: Internal escape hatch — if provided (Qwen3 wrapper passes the
            tuple it already gathered), short-circuits arch-specific gather. New callers
            should pass `arch=` only.
        mrope_section: Required for arch='qwen3_vl_text';
            rejected for non-mrope archs. Forwarded as the trailing append-only
            kwarg of `causal_decoder_set_weights`.
        cos_sin: Override for the kernel-format cos/sin tables `[max_seq, head_dim/2]`.
            Required for mrope (kernel rejects HF-expanded `[max_seq, head_dim]`).
            None → fall back to rotary_emb introspection (legacy 1D-RoPE).

    Returns:
        int: The handle for this model. Also stored as `model._rpu_decoder_handle`.
    """
    # ``qwen3_vl_text`` is the M-RoPE-enabled variant.
    if arch not in ("qwen3", "llama", "qwen3_vl_text"):
        raise ValueError(
            f"_install_causal_decoder_forward: arch must be 'qwen3', 'llama', "
            f"or 'qwen3_vl_text', got {arch!r}"
        )
    if arch == "qwen3_vl_text":
        if mrope_section is None or len(mrope_section) == 0:
            raise ValueError(
                "_install_causal_decoder_forward(arch='qwen3_vl_text'): "
                "mrope_section must be provided "
                "(read from text_config.rope_parameters['mrope_section'])"
            )
    else:
        if mrope_section is not None and len(mrope_section) > 0:
            raise ValueError(
                f"_install_causal_decoder_forward(arch={arch!r}): "
                f"mrope_section must be empty for non-mrope archs"
            )

    # Validate `arch` against model.config.model_type to catch caller-side
    # architecture/config mismatches. Qwen3-VL text decoder shares the qwen3
    # weight layout (same QKV / RMSNorm / SwiGLU pattern); allow it through.
    _ARCH_MAP = {"qwen3": "qwen3", "llama": "llama", "qwen3_vl_text": "qwen3_vl_text"}
    cfg_type = getattr(getattr(model, "config", None), "model_type", None)
    expected_arch = _ARCH_MAP.get(cfg_type) if cfg_type is not None else None
    if expected_arch is not None and expected_arch != arch:
        raise ValueError(
            f"_install_causal_decoder_forward: arch={arch!r} mismatches "
            f"model.config.model_type={cfg_type!r} (expected arch={expected_arch!r}). "
            "Likely cause: an adapter is wired to the wrong arch path; check "
            "adapters/{qwen3,llama}.py:patch_*_for_causal_lm delegation."
        )

    try:
        from transformers.modeling_outputs import BaseModelOutputWithPast
    except ImportError as e:
        raise RuntimeError(
            "_install_causal_decoder_forward requires transformers with BaseModelOutputWithPast"
        ) from e

    # Internal escape hatch: if caller already gathered qk_norm_lists, use it; otherwise
    # gather based on arch.
    if qk_norm_lists is None:
        if arch in ("qwen3", "qwen3_vl_text"):
            num_layers_local = len(model.layers)
            if num_layers_local == 0:
                raise RuntimeError(
                    f"_install_causal_decoder_forward(arch={arch!r}): model has no layers"
                )
            q_norm_list = [model.layers[i].self_attn.q_norm.weight for i in range(num_layers_local)]
            k_norm_list = [model.layers[i].self_attn.k_norm.weight for i in range(num_layers_local)]
            qk_norm_lists = (q_norm_list, k_norm_list)
        else:  # llama
            # Body materializes to [], [] below; empty Python lists are valid
            # Tensor[] at the C++ boundary.
            qk_norm_lists = None

    # Keep the currently published install alive until its replacement is
    # completely prepared. Direct adapter callers use this re-entry path for
    # lm-head fuse changes; destroying the old handle up front would turn a
    # failed replacement into a half-installed model.
    _missing = object()
    old_handle = getattr(model, "_rpu_decoder_handle", _missing)
    old_finalizer = getattr(model, "_rpu_decoder_handle_finalizer", None)
    old_install_state = {
        attr: getattr(model, attr, _missing)
        for attr in _CAUSAL_DECODER_INSTALL_ATTRS
    }
    model_state = vars(model)
    had_instance_forward = "forward" in model_state
    old_instance_forward = model_state.get("forward")

    # 1. Gather per-layer weights + global params + final_norm_w.
    # No native handle or public model state is mutated during this phase.
    num_layers = len(model.layers)
    if num_layers == 0:
        raise RuntimeError("_install_causal_decoder_forward: model has no layers")

    def _layer_weight(i, path):
        layer = model.layers[i]
        obj = layer
        for attr in path:
            obj = getattr(obj, attr)
        return obj.weight

    q_w_list          = [_layer_weight(i, ["self_attn", "q_proj"])      for i in range(num_layers)]
    k_w_list          = [_layer_weight(i, ["self_attn", "k_proj"])      for i in range(num_layers)]
    v_w_list          = [_layer_weight(i, ["self_attn", "v_proj"])      for i in range(num_layers)]
    o_w_list          = [_layer_weight(i, ["self_attn", "o_proj"])      for i in range(num_layers)]
    if qk_norm_lists is not None:
        q_norm_list, k_norm_list = qk_norm_lists
    else:
        q_norm_list, k_norm_list = [], []  # empty lists select the no-QK-norm path
    input_norm_list   = [model.layers[i].input_layernorm.weight         for i in range(num_layers)]
    post_norm_list    = [model.layers[i].post_attention_layernorm.weight for i in range(num_layers)]
    gate_list         = [_layer_weight(i, ["mlp", "gate_proj"])         for i in range(num_layers)]
    up_list           = [_layer_weight(i, ["mlp", "up_proj"])           for i in range(num_layers)]
    down_list         = [_layer_weight(i, ["mlp", "down_proj"])         for i in range(num_layers)]

    attn0 = model.layers[0].self_attn
    head_dim = attn0.head_dim
    num_q_heads = attn0.q_proj.out_features // head_dim
    num_kv_heads = attn0.k_proj.out_features // head_dim
    hidden_size = attn0.o_proj.out_features
    intermediate_size = model.layers[0].mlp.gate_proj.out_features
    eps = model.layers[0].input_layernorm.variance_epsilon

    # cos/sin override: callers (e.g. Qwen3-VL text) can pre-compute the
    # kernel-format [max_seq, head_dim/2] tables themselves and bypass the
    # rotary_emb introspection below. Required for mrope: the cached/computed
    # rotary forward on Qwen3-VL returns the HF-expanded form, which the mrope
    # kernel rejects.
    if cos_sin is not None:
        cos_cached, sin_cached = cos_sin
    else:
        rotary = getattr(model, "rotary_emb", None)
        if rotary is None:
            rotary = getattr(model.layers[0].self_attn, "rotary_emb", None)
        if rotary is None:
            raise RuntimeError(
                "_install_causal_decoder_forward: could not locate rotary embedding module "
                "(tried model.rotary_emb and layers[0].self_attn.rotary_emb)"
            )

        cos_cached = getattr(rotary, "cos_cached", None)
        sin_cached = getattr(rotary, "sin_cached", None)
        if cos_cached is None or sin_cached is None:
            device = next(model.parameters()).device
            # Resolve every supported rotary capacity spelling and keep the
            # largest authoritative value.
            max_pos = _rotary_table_capacity(rotary, model)
            dummy = torch.zeros(1, 1, hidden_size, dtype=torch.float16, device=device)
            dummy_pos = torch.arange(max_pos, device=device).unsqueeze(0)
            cos_cached, sin_cached = rotary(dummy, dummy_pos)
            if cos_cached.dim() == 3:
                cos_cached = cos_cached.squeeze(0)
                sin_cached = sin_cached.squeeze(0)

    cos_cached = cos_cached.to(dtype=torch.float16, device="rpu").contiguous()
    sin_cached = sin_cached.to(dtype=torch.float16, device="rpu").contiguous()

    final_norm_w = model.norm.weight  # RMSNorm final projection

    use_silu = True  # Qwen3 / Llama / Phi / Mistral all use SwiGLU (SiLU(gate) * up)

    # Pass mrope_section as the trailing append-only argument. Empty means
    # standard one-dimensional RoPE.
    mrope_section_arg = list(mrope_section) if mrope_section else []
    # The same append-only contract applies to DeepStack language layers.
    # Empty means no DeepStack injection.
    deepstack_lang_layers_arg = (
        [int(x) for x in deepstack_lang_layers] if deepstack_lang_layers else []
    )
    has_w8a16_scales = scale_lists is not None
    if has_w8a16_scales:
        q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws = scale_lists
    else:
        q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws = [], [], [], [], [], [], []
    set_weights_op = (
        torch.ops.rpu.causal_decoder_set_weights_w8a16
        if has_w8a16_scales
        else torch.ops.rpu.causal_decoder_set_weights
    )
    set_weights_args = [
        q_w_list, k_w_list, v_w_list, o_w_list,
        q_norm_list, k_norm_list,
        input_norm_list, post_norm_list,
        gate_list, up_list, down_list,
        cos_cached, sin_cached,
        final_norm_w,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size,
        eps, use_silu,
        mrope_section_arg,
        deepstack_lang_layers_arg,
    ]
    if has_w8a16_scales:
        set_weights_args.extend([
            q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws,
        ])

    # Create a per-instance GraphCache so the patched forward can wrap
    # `_run_causal_decoder_forward` in `with graph_cache.capture(sig):`.
    # Without the wrap, every operator uses passthrough submission.
    #
    # The Qwen3-VL e2e flow (adapters/qwen3_vl/__init__.py) bypasses this
    # patched forward and uses its own `_rpu_text_graph_cache` external
    # wrap; the cache stashed here stays dormant for that path (small
    # one-off memory cost — no second BUILD because forward never enters
    # this closure).
    from contextlib import nullcontext as _nullcontext
    import rpu_backend as _rb
    decoder_graph_cache = _rb.graph.GraphCache()
    decoder_batch_decode_enabled = arch == "qwen3"
    decoder_num_layers = int(num_layers)
    decoder_hidden_size = int(hidden_size)
    # FNV1a-style tiebreaker prevents plain Qwen3 and Qwen3-VL DeepStack
    # signatures from aliasing on one model instance.
    _ds_hash = 0
    for _idx in deepstack_lang_layers_arg:
        _ds_hash = (_ds_hash * 1099511628211) ^ int(_idx)
        _ds_hash &= (1 << 63) - 1
    _GraphSignature = _rb.graph.GraphSignature  # captured by closure below
    from rpu_backend.api.cache import RPUCache as _RPUCache_for_sig

    # 3. Replace the instance's forward method with our all-layers-once version.
    def rpu_decoder_model_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        # Only build a capture sig when inputs are valid — otherwise let
        # `_run_causal_decoder_forward` raise its precise AssertionError
        # (past_key_values not RPUCache, or input_ids/inputs_embeds XOR
        # violation). `past_key_values.position` would otherwise blow up here.
        prefill_plan = None
        batched_prefill = _is_batched_prefill(input_ids, inputs_embeds)
        if isinstance(past_key_values, _RPUCache_for_sig):
            if inputs_embeds is not None:
                _batch, _seq_len = (int(inputs_embeds.shape[0]),
                                    int(inputs_embeds.shape[1]))
            elif input_ids is not None:
                _batch, _seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
            else:
                _batch, _seq_len = 1, 0  # runner will raise on XOR violation
            if arch != "qwen3_vl_text" and not batched_prefill:
                attention_mask, position_ids, cache_position = (
                    _canonicalize_plain_text_controls(
                        _seq_len,
                        past_key_values,
                        attention_mask,
                        position_ids,
                        cache_position,
                        batch_size=_batch,
                    )
                )
            else:
                # CPU → RPU prep MUST happen outside graph capture. Qwen3-VL
                # keeps semantic M-RoPE positions; batched Qwen3 validates and
                # removes its no-op positions in the shared helper.
                if (attention_mask is not None
                        and attention_mask.device.type != "rpu"):
                    attention_mask = attention_mask.to("rpu")
                if (not batched_prefill
                        and position_ids is not None
                        and position_ids.device.type != "rpu"):
                    position_ids = position_ids.to("rpu")
            if not batched_prefill:
                _reject_unconsumed_text_padding(
                    self,
                    _seq_len,
                    attention_mask,
                    position_ids,
                )
            _execution_len = _seq_len
            _planned_chunk_size = 0
            if (_seq_len > 1
                    and position_ids is None
                    and (attention_mask is None or batched_prefill)):
                prefill_plan = _text_prefill_execution_plan(
                    self,
                    self._rpu_decoder_handle,
                    past_key_values,
                    _seq_len,
                )
                _execution_len, _planned_chunk_size = prefill_plan
            # Position is omitted from the decode signature so successive steps
            # hit the same cached entry (first step BUILD, rest REPLAY through
            # the sync-only fast path). This is safe because every position-derived
            # kernel arg flows through set_regs + add_kernel_mutable and is
            # refreshed by sync_mutable_params on each REPLAY.
            # Batch, logical length and physical execution plan all key the
            # graph: each changes either dispatch count or baked SPM offsets.
            def _capture(batch, seq_len, plan=None):
                execution_len, planned_chunk_size = (
                    (int(plan[0]), int(plan[1]))
                    if plan is not None else (int(seq_len), 0)
                )
                return self._rpu_decoder_graph_cache.capture(_GraphSignature(
                    op_id="rpu_causal_decoder",
                    shapes=[
                        int(batch),
                        int(seq_len),
                        execution_len,
                        self._rpu_decoder_hidden_size,
                    ],
                    dyn_dims=[
                        self._rpu_decoder_num_layers,
                        self._rpu_decoder_deepstack_hash,
                        chunk_policy_key(self._rpu_decoder_handle),
                        planned_chunk_size,
                        prefill_position_key(
                            execution_len, past_key_values.position
                        ),
                    ],
                    dtypes=[torch.float16],
                ))
        else:
            # Preserve the runner's precise invalid-cache/input error while
            # keeping any CPU→RPU transfer outside capture.
            if attention_mask is not None and attention_mask.device.type != "rpu":
                attention_mask = attention_mask.to("rpu")
            if position_ids is not None and position_ids.device.type != "rpu":
                position_ids = position_ids.to("rpu")
            def _capture(batch, seq_len, plan=None):
                return _nullcontext()
            _batch, _seq_len = 1, 0

        if _batch > 1 and not bool(self._rpu_batch_decode_enabled):
            raise AssertionError(
                "RPU batch > 1 is enabled only for the Qwen3 decoder "
                "capability"
            )

        _runner_kwargs = dict(
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        if _is_batched_prefill(input_ids, inputs_embeds):
            # [B>1, S>1] cannot go through the kernel in one call — drive it as
            # B captured single-sequence prefills, one KV slot each.
            raw, pkv = _run_batched_prefill(
                self, self._rpu_decoder_handle, _capture,
                past_key_values=past_key_values,
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                prefill_plan=prefill_plan,
                **_runner_kwargs,
            )
        else:
            with _capture(_batch, _seq_len, prefill_plan):
                raw, pkv = _run_causal_decoder_forward(
                    self,
                    self._rpu_decoder_handle,
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    prefill_plan=prefill_plan,
                    **_runner_kwargs,
                )

        # Base-model contract: raw MUST be [batch, seq, hidden_size].
        _validate_decoder_base_output(
            raw, self.config.hidden_size
        )

        return BaseModelOutputWithPast(
            last_hidden_state=raw,
            past_key_values=pkv,
            hidden_states=None,
            attentions=None,
        )

    # 2. Configure and tentatively publish the replacement while retaining a
    # complete snapshot of the old Python state. The old native handle remains
    # live until every Python assignment succeeds. Publication or old-handle
    # retirement failure restores the snapshot and immediately finalizes the
    # pending handle.
    handle = torch.ops.rpu.causal_decoder_create()
    handle_finalizer = None
    committed = False
    try:
        handle_finalizer = weakref.finalize(
            model, _destroy_causal_decoder_handle, handle
        )
        set_weights_op(handle, *set_weights_args)
        # Declare the certified chunk envelope before the first forward.
        # Deny-by-default — the C++ planner refuses to prefill an undeclared
        # handle, because the auto search can otherwise pick a chunk above the
        # model's verified SPM ceiling. The caller owns the architecture-specific
        # table so this runtime module stays architecture-independent.
        _envelope = chunk_envelope_for(
            arch, decoder_num_layers, decoder_hidden_size)
        torch.ops.rpu.causal_decoder_set_chunk_envelope(
            handle, int(_envelope[0]), int(_envelope[1]))

        execution_config = (
            {} if execution_config is None else execution_config
        )
        _prefill_cfg = execution_config.get("prefill", {})
        _requested_chunk = _prefill_cfg.get("chunk_size", "auto")
        torch.ops.rpu.causal_decoder_set_chunk_size_override(
            handle,
            0 if _requested_chunk == "auto" else int(_requested_chunk),
        )

        model._rpu_deepstack_lang_layers = deepstack_lang_layers_arg
        model._rpu_batch_decode_enabled = decoder_batch_decode_enabled
        model._rpu_decoder_graph_cache = decoder_graph_cache
        model._rpu_decoder_num_layers = decoder_num_layers
        model._rpu_decoder_hidden_size = decoder_hidden_size
        model._rpu_decoder_deepstack_hash = _ds_hash
        model._rpu_decoder_handle = handle
        model._rpu_decoder_handle_finalizer = handle_finalizer
        model._rpu_prefill_execution_alignment = (
            16 if arch == "qwen3_vl_text" else 1
        )
        model._rpu_execution = execution_config

        import types
        model.forward = types.MethodType(rpu_decoder_model_forward, model)

        if (
            old_handle is not _missing
            and (
                old_finalizer is None
                or getattr(old_finalizer, "alive", False)
            )
        ):
            torch.ops.rpu.causal_decoder_destroy(old_handle)
        committed = True
    finally:
        if not committed:
            # Rollback uses the instance dictionary deliberately: a failing
            # custom hardware-attribute __setattr__ is one of the publication failures
            # this path must recover from.
            state = vars(model)
            for attr, old_value in old_install_state.items():
                if old_value is _missing:
                    state.pop(attr, None)
                else:
                    state[attr] = old_value
            if had_instance_forward:
                state["forward"] = old_instance_forward
            else:
                state.pop("forward", None)

            if handle_finalizer is None:
                _destroy_causal_decoder_handle(handle)
            elif handle_finalizer.alive:
                handle_finalizer()

    # The old callback must not outlive its retired handle. The current native
    # registry uses monotonic IDs, but detaching still preserves one callback
    # per handle and avoids a redundant not-found destroy at model GC.
    if old_finalizer is not None and getattr(old_finalizer, "alive", False):
        old_finalizer.detach()

    _LOG.info("Patched %s (instance) with all-layers-once fused forward, "
              "handle=%d, num_layers=%d",
              type(model).__name__, handle, num_layers)

    return handle
