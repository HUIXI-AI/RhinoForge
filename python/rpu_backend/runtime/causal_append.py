"""Adapter-owned causal continuation using existing prefill/decode calls.

This does not change native chunk admission.  Explicit chunk/padding requests
remain one native plan; only an unpadded auto request may be decomposed.
"""
from __future__ import annotations

import copy

import torch


def continuation_segments(position: int, length: int, stage, *, capacity: int):
    """Return logical (offset, length) spans, or None for the normal planner.

    Decode consumes a non-aligned prefix/tail without inventing cached tokens.
    The aligned body must still pass the real native planner before execution.
    """
    position, length, capacity = int(position), int(length), int(capacity)
    if position < 0 or length < 1 or position + length > capacity:
        raise ValueError("causal continuation exceeds the logical KV-cache horizon")
    if position == 0 or length == 1 or stage.get("chunk_size", "auto") != "auto":
        return None
    padding = stage.get("padding_rows", "auto")
    if padding not in (0, "auto"):
        return None
    # An auto-padding request retains its bounded native search.  An explicit
    # zero rows request has no search budget, even if the caller supplied one.
    if padding != 0 and int(stage.get("padding_budget", 64)) != 0:
        return None
    if position % 16 == 0 and length % 16 == 0:
        return None
    spans = []
    head = min((-position) % 16, length)
    spans.extend((offset, 1) for offset in range(head))
    body = ((length - head) // 16) * 16
    if body:
        spans.append((head, body))
    spans.extend((offset, 1) for offset in range(head + body, length))
    return tuple(spans)


def run_continuation_segments(forward, cache, owner, spans, arguments):
    """Compose top-level logits while retaining every native execution receipt.

    The caller must validate all semantic inputs and preflight every span first.
    Each forward owns its usual capture and lm_head path.  No raw hidden states
    are combined with decode-fused logits.
    """
    start = int(cache.position)
    length = sum(count for _, count in spans)
    keep = arguments.get("logits_to_keep", 0)
    last_only = isinstance(keep, int) and keep == 1
    logits, receipts = [], []
    owner_state = vars(owner)
    had_receipt = "_rpu_last_execution_plan" in owner_state
    previous_receipt = owner_state.get("_rpu_last_execution_plan")
    try:
        for offset, count in spans:
            call = dict(arguments)
            for name in ("input_ids", "inputs_embeds"):
                if call.get(name) is not None:
                    call[name] = call[name][:, offset:offset + count].contiguous()
            if call.get("attention_mask") is not None:
                call["attention_mask"] = call["attention_mask"][..., offset:offset + count].contiguous()
            if call.get("cache_position") is not None:
                call["cache_position"] = call["cache_position"].reshape(-1)[offset:offset + count].contiguous()
            if call.get("position_ids") is not None:
                value = call["position_ids"]
                call["position_ids"] = (
                    value[offset:offset + count] if value.ndim == 2 and value.shape[-1] == 3
                    else value[..., offset:offset + count]
                ).contiguous()
            call["logits_to_keep"] = 1 if last_only else 0
            output = forward(**call)
            if int(cache.position) != start + offset + count:
                raise RuntimeError("causal continuation segment advanced the wrong logical position")
            receipt = getattr(owner, "_rpu_last_execution_plan", None)
            if not isinstance(receipt, dict) or receipt.get("logical_len") != count or receipt.get("position") != start + offset:
                raise RuntimeError("causal continuation segment has no matching execution receipt")
            receipts.append(copy.deepcopy(receipt))
            # Later dispatches may reuse native output storage.  Keep only the
            # final output for last-token mode and own every retained row.
            if not last_only or offset + count == length:
                logits.append(output.logits.clone())
        joined = logits[0] if len(logits) == 1 else torch.cat(logits, dim=1)
        if not last_only:
            selected = slice(-keep, None) if isinstance(keep, int) and keep else (
                slice(None) if isinstance(keep, int) else keep)
            joined = joined[:, selected, :]
        output.logits = joined
    except BaseException:
        # All writes are at or after the original logical end; rewinding hides
        # a partially written append and the next request overwrites it.
        try:
            cache.reset_to_position(start)
        finally:
            # A completed child publishes its receipt before a later child can
            # fail.  Restore the pre-call evidence together with the logical KV
            # position so a rejected append cannot look partially committed.
            if had_receipt:
                owner_state["_rpu_last_execution_plan"] = previous_receipt
            else:
                owner_state.pop("_rpu_last_execution_plan", None)
        raise
    # This is adapter-owned evidence, not a public hardware configuration
    # mutation. Validated Qwen3 modules reject late setattr of internal names;
    # native stage publication uses the same direct instance-state route.
    owner_state["_rpu_last_execution_plan"] = {
        "stage": "prefill", "component": receipts[-1].get("component", "language_model"),
        "generation": receipts[-1].get("generation", 0),
        "position": start, "logical_len": length,
        "execution_len": sum(row["execution_len"] for row in receipts),
        "padding_rows": sum(row.get("padding_rows", 0) for row in receipts),
        "chunk_size": None,
        "execution_kind": "segmented_append",
        "authority": "ADAPTER_SEGMENTED_NATIVE_PLANS",
        "selection_scope": "UNPADDED_CAUSAL_CONTINUATION",
        "segments": receipts,
    }
    return output
