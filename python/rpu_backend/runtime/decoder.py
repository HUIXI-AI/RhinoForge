"""rpu_backend.runtime.decoder — generic decoder helpers.

Per ADR §6.4 + REQ V2D-06 + V2D-05: private helpers consumed by adapters/<arch>.py.
This module is INTERNAL — adapters call these directly.

v5-04 lift: byte-equal copy of `patch_rmsnorm_class` + `patch_rotary_embedding`
from `_internal/patches/__init__.py`. Renamed to `_install_rmsnorm_class_swap`
+ `_install_rotary_class_swap` to make the side-effecting "install once"
semantics explicit at the call site.

v5-11 (Phase 11) NS-03 closure adds 3 more helpers absorbed from
`_internal/patches/__init__.py` (deleted in this phase):
  - `_install_causal_decoder_forward` (was `patch_causal_decoder_for_rpu` at :393)
  - `_run_causal_decoder_forward` (was `_rpu_qwen3_run_forward` at :277; renamed
    because the implementation is shared
    cross-arch, not Qwen3-specific; "qwen3" in old name was historical misnomer)
  - `chunk_policy_key` (MR-D; replaced `_get_chunk_size`, which read the
    process-global chunk size that MR-D deleted)

ADR §3.2 DAG: this module imports only from `runtime.*` and `api.*` (downward).
adapters/* depends on this module (downward); runtime/decoder.py MUST NOT import
from adapters/*.
"""
from __future__ import annotations

import json
import os
import sys
import torch
from typing import Any, Type

from rpu_backend.runtime.log import _LOG
from rpu_backend.runtime.control import rpu_env_bool
from rpu_backend.runtime._native_retirement import _InstalledNativeResource
from rpu_backend.runtime.topology import (
    execution_core_count, resolve_decoder_topology, decoder_mlp_intermediate_size,
    require_same_decoder_topology, validate_decoder_cache_topology,
)
from rpu_backend.runtime.execution_planner import (
    GRAPH_RETAINED_CACHE,
    PlannerRejectError,
    native_chunk_reject,
    plan_prefill,
)


# patch-reason: (a) RMSNorm kernel dispatch — §3a (a) run-different-op-on-RPU
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
            # Use native PyTorch operator dispatch for the RPU RMSNorm kernel.
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


# patch-reason: (a) RoPE class swap — §3a (a) run-different-op-on-RPU
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
                # The tensors are tiny ([B, seq_len, head_dim], ~few KB) so the
                # per-forward CPU cost is negligible vs the rotary apply kernel.
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


# ═════════════════════════════════════════════════════════════════════════
# v5-11 NS-03 closure: 3 helpers absorbed from _internal/patches/__init__.py
# The causal forward and chunk-size helpers belong to the runtime layer.
# _run_causal_decoder_forward and the chunk-size helper are
# co-located here (NOT in adapters/qwen3.py per original D-03a/b) to avoid a
# runtime → adapters DAG violation.
# ═════════════════════════════════════════════════════════════════════════


def chunk_policy_key(handle: int) -> int:
    """This handle's chunk override, to be folded into its GRAPH SIGNATURE.

    MR-D. The chunk plan decides the SPM layout — ctx.chunk_size feeds
    compute_params_hash_impl, so a policy change takes ensure_allocated's Path 1
    and re-derives every temp offset. A graph captured under one policy has that
    policy's offsets baked into its DMA descriptors, so two policies at one shape
    MUST be two signatures or the second forward replays the first one's graph
    under a layout it was not recorded for ("DMA bytes drift at cursor N").

    Why the override alone is enough, and why it is not the RESOLVED chunk:
      - under an explicit override, alloc_ctx.chunk_size IS the override
        (fused_model_base.cpp:742), so this is exactly the discriminator;
      - under auto (0), the resolved chunk is a function of the shape — already
        in the signature — and of the cap/envelope, which are frozen before the
        first forward (their setters check get_last_resolved_chunk_size() == 0).
    Calling the planner instead would be strictly more expensive for the same
    partition, and this has to be affordable per decode step, where the override
    reaches alloc_ctx.chunk_size just as it does in prefill.

    ⚠️ The "function of the shape" clause above is INCOMPLETE. See
    `prefill_position_key()` below for the half it omits.

    (Replaced `_get_chunk_size()`, which read the process-global chunk size that
    MR-D deleted. It was documented dead — no production callers — and its global
    is gone, so it went with it.)
    """
    return int(torch.ops.rpu.causal_decoder_get_chunk_size_override(handle))


def prefill_position_key(seq_len: int, position: int) -> int:
    """The resume position of a MULTI-TOKEN forward, for its graph signature.

    `chunk_policy_key` claims the resolved chunk is, under auto, a function of
    the shape. It is not: chunk validity depends on the KV CONTEXT LENGTH, which
    is `position + execution_len`, not `execution_len`. `rpu_qwen3_model.h:120`
    says so in as many words — "a plan resolved at position 0 can be REJECTED by
    compute_chunks at position P". So one shape at two resume positions can
    resolve to two chunk sizes, i.e. two SPM layouts, and the second forward
    replays the first one's graph under a layout it was never recorded for.

    The signature must key on the input to that decision even when two
    positions happen to resolve to the same layout for a particular model.

    DECODE IS DELIBERATELY EXEMPT (`seq_len <= 1` -> 0). P3 dropped position from
    this signature so successive decode steps share one entry, and that reasoning
    still holds there: every position-derived KERNEL ARG is mutable and refreshed
    per REPLAY. What is not mutable is the SPM layout — and at seq_len=1 the
    search range is [16, 16], so the layout cannot move. Keying decode on
    position would put one graph per generated token in the cache, which is the
    regression this exemption exists to prevent.

    Prefill at position 0 — every single-turn caller — still hashes to 0, so no
    existing signature changes and no cache entry is added.
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
    "_execution_session",
    "_rpu_deepstack_lang_layers",
    "_rpu_batch_decode_enabled",
    "_rpu_decoder_graph_cache",
    "_rpu_decoder_num_layers",
    "_rpu_decoder_hidden_size",
    "_rpu_decoder_deepstack_hash",
    "_rpu_decoder_topology",
    "_rpu_decode_stage_descriptor",
    "_decoder_decode_plan",
    "_rpu_decoder_handle",
    "_rpu_decoder_handle_finalizer",
    "_rpu_decoder_retirement_state",
    "_rpu_prefill_execution_alignment",
    "_rpu_kv_cache_layer_bank_size",
    "_rpu_execution",
    "_rpu_execution_generation",
    "_rpu_last_execution_plan",
)

def _bind_causal_decoder_execution_session(adapter, *, entry_point: str):
    """Bind one retained-cache control session to a plain CausalLM owner."""
    from rpu_backend.api._execution import (
        _CAUSAL_DECODER_EXECUTION_SUPPORTED,
        _cold_causal_decoder_execution,
        bind_execution_session,
        bind_rpu_execution,
        native_execution_reconfigure,
    )

    model = adapter.model
    inner = model.model
    supported = getattr(adapter, "EXECUTION_SUPPORTED", _CAUSAL_DECODER_EXECUTION_SUPPORTED)
    execution = bind_rpu_execution(
        model,
        getattr(model, "_rpu_execution", None),
        entry_point=entry_point,
        supported=supported,
    )
    existing_session = getattr(model, "_execution_session", None)
    if existing_session is None:
        execution = _cold_causal_decoder_execution(execution)
    journal = {}

    def native_chunk(config) -> int:
        value = config.get("prefill", {}).get("chunk_size", "auto")
        return 0 if value == "auto" else int(value)

    def reset_graphs() -> None:
        cache = getattr(inner, "_rpu_decoder_graph_cache", None)
        if cache is None:
            raise RuntimeError(
                f"{entry_point}: live decoder has no retained GraphCache"
            )
        cache.begin_warmup()
        cache.clear()
        if not cache.cache_invariant_ok():
            raise RuntimeError(
                f"{entry_point}: GraphCache invariant failed during "
                "execution reconfigure"
            )
        state = vars(inner)
        state.pop("_rpu_last_a6_plan", None)
        state.pop("_rpu_last_execution_plan", None)

    def validate(config) -> None:
        require_same_decoder_topology(model._execution_session.config, config)
        if (bool(config.get("prefill", {}).get("linear_acc32", False)) !=
                bool(model._execution_session.config.get("prefill", {}).get("linear_acc32", False))):
            raise ValueError("linear_acc32 is cold-only; close and reload the model")
        if not _causal_lm_runtime_complete(model):
            raise RuntimeError(
                f"{entry_point}: reconfigure requires to_rpu() to complete first"
            )
        adapter.preflight_execution(model.config, config)
        required = (
            "execution_reconfigure_begin",
            "execution_reconfigure_commit",
            "execution_reconfigure_abort",
            "execution_reconfigure_abort_attempt",
            "causal_decoder_stage_chunk_size_override",
        )
        missing = [name for name in required if not hasattr(torch.ops.rpu, name)]
        if missing:
            raise RuntimeError(
                f"{entry_point}: binary lacks native execution-reconfigure "
                "op(s): " + ", ".join(missing)
            )

    def apply(old, new, generation: int, *, force_rebuild=False) -> None:
        journal.clear()
        journal.update({
            "mutation_started": False,
            "graphs_touched": False,
            "old_generation": int(
                getattr(inner, "_rpu_execution_generation", generation - 1)
            ),
        })
        old_chunk = native_chunk(old)
        new_chunk = native_chunk(new)
        if old_chunk != new_chunk:
            with native_execution_reconfigure(torch.ops.rpu) as token:
                journal["mutation_started"] = True
                torch.ops.rpu.causal_decoder_stage_chunk_size_override(
                    inner._rpu_decoder_handle, token, new_chunk
                )

        journal["graphs_touched"] = True
        reset_graphs()
        vars(inner)["_rpu_execution"] = new
        vars(inner)["_rpu_execution_generation"] = int(generation)
        vars(model)["_rpu_execution_generation"] = int(generation)

    def rollback(old, _new, generation: int) -> None:
        if journal.get("mutation_started", False):
            with native_execution_reconfigure(torch.ops.rpu) as token:
                torch.ops.rpu.causal_decoder_stage_chunk_size_override(
                    inner._rpu_decoder_handle, token, native_chunk(old)
                )
        if journal.get("graphs_touched", False):
            reset_graphs()
        old_generation = int(journal.get("old_generation", generation))
        vars(inner)["_rpu_execution"] = old
        vars(inner)["_rpu_execution_generation"] = old_generation
        vars(model)["_rpu_execution_generation"] = old_generation
        journal.clear()

    callbacks = {}
    if existing_session is None:
        callbacks = {
            "validate": validate,
            "apply": apply,
            "rollback": rollback,
        }
    session = bind_execution_session(
        model,
        execution,
        entry_point=entry_point,
        supported=supported,
        graph_mode="RETAINED_CACHE",
        **callbacks,
    )
    # The public loader returns ``model``. Keep that owner, its adapter facade,
    # and the physical inner decoder on the exact same immutable generation.
    vars(model)["_rpu_execution"] = session.config
    session.register_config_view(adapter)
    session.register_config_view(inner)
    vars(inner)["_execution_session"] = session
    # Cold construction owns the generation on the session. Publishing private
    # runtime metadata before the public-only preinstall scan makes a clean
    # Qwen3/Llama model look like a partially installed or user-modified model.
    # The installer binds again once its native decoder exists; publish both
    # runtime mirrors then, without admitting arbitrary private cold attrs.
    if getattr(inner, "_rpu_decoder_handle", None) is not None:
        vars(model)["_rpu_execution_generation"] = int(session.generation)
        vars(inner)["_rpu_execution_generation"] = int(session.generation)
    adapter._execution_session = session
    return session


def _enable_causal_decoder_execution_reconfigure(inner) -> None:
    enable = getattr(
        torch.ops.rpu, "causal_decoder_enable_execution_reconfigure", None
    )
    if enable is None:
        raise RuntimeError(
            "RPU binary lacks causal-decoder hot-reconfigure support"
        )
    enable(inner._rpu_decoder_handle)


def _new_decoder_graph_cache(topology=None, *, max_entries=None):
    """Bind graph queue/snapshot resources to the decoder's cold core budget."""
    from rpu_backend.graph import GraphCache, GraphRuntimePolicy

    capacity = {"max_entries": max_entries} if max_entries is not None else {}
    if topology is None:
        return GraphCache(**capacity)
    return GraphCache(runtime_policy=GraphRuntimePolicy.from_environment(
        execution_core_count=topology.num_cores), **capacity)


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
        and bool(getattr(inner, "_rpu_decode_stage_descriptor", ()))
        and getattr(installed_forward, "__self__", None) is inner
        and getattr(forward_function, "__name__", None)
        == "rpu_decoder_model_forward"
    )


def _close_causal_lm_model(model) -> None:
    """Explicit inference teardown shared by public examples and v5."""
    from rpu_backend.api.causal_lm import _release_live_instance

    gemma4 = getattr(model, "_rpu_gemma4_retirement_state", None)
    if gemma4 is not None or getattr(model, "_rpu_gemma4_handle", None) is not None:
        if gemma4 is None:
            raise RuntimeError("Gemma4 teardown requires its actual retirement state")
        gemma4.retire_owned(model)
        for attr in ("_rpu_gemma4_handle", "_rpu_gemma4_handle_finalizer",
                     "_rpu_gemma4_retirement_state"):
            vars(model).pop(attr, None)
        _release_live_instance(model)
    inner = getattr(model, "model", None)
    if getattr(inner, "_rpu_decoder_handle", None) is not None:
        # Del/GC is not retirement evidence and intentionally swallows errors.
        def retire():
            inner._rpu_decoder_retirement_state.retire()
            for attr in ("_rpu_decoder_handle", "_rpu_decoder_handle_finalizer",
                         "_rpu_decoder_graph_cache", "_rpu_decoder_retirement_state"):
                vars(inner).pop(attr, None)

        model._execution_session.shutdown(retire)
        if hasattr(inner, "_rpu_decoder_handle"):
            raise RuntimeError("CausalLM session closed without retiring its handle")
        _release_live_instance(model)


def _cleanup_causal_decoder_install(
    inner,
    *,
    had_instance_forward: bool,
    original_instance_forward,
) -> None:
    """Retire an outer failed install without hiding its original exception."""
    if inner is None:
        return
    resource = getattr(inner, "_rpu_decoder_retirement_state", None)
    if resource is None and getattr(inner, "_rpu_decoder_handle", None) is not None:
        raise RuntimeError("decoder cleanup requires its actual retirement state")
    if resource is not None:
        error = sys.exception()
        session = getattr(inner, "_execution_session", None)
        try:
            if session is not None:
                session.shutdown(resource.retire)
            else:
                resource.retire()
            if resource.handle is not None:
                raise RuntimeError("decoder session closed without retiring its handle")
        except BaseException as cleanup_error:
            resource.retain_failure(cleanup_error, inner)
            if error is None:
                raise
            error.add_note(f"decoder cleanup failed: {cleanup_error!r}")
            return

    # An outer publication failure can activate the model's A9/custom
    # attribute guard before cleanup runs. Restore plain install metadata
    # without re-entering those hooks.
    inner_state = vars(inner)
    if had_instance_forward:
        inner_state["forward"] = original_instance_forward
    else:
        inner_state.pop("forward", None)
    for attr in _CAUSAL_DECODER_INSTALL_ATTRS:
        inner_state.pop(attr, None)


def _mint_planner_calibration_candidate(native, execution_len, logical_len,
                                        position, graph_mode, row, route):
    """Decode the real owner's admitted EXACT result while its oracle is live."""
    from rpu_backend.runtime.execution_planner import StageTuple, _decode_native_stage_domain

    descriptor, layout_hi, layout_lo = getattr(
        torch.ops.rpu, native[0] + "_kvinsert_exact_candidate")(
            native[1], list(row.physical_descriptor), *route)
    if any(type(value) is not int or not 0 <= value <= 0xFFFFFFFF
           for value in (layout_hi, layout_lo)) or not (layout_hi or layout_lo):
        raise ValueError("native calibration requires an actual nonzero layout identity")
    decoded, = _decode_native_stage_domain(
        (1, 1, len(descriptor), *descriptor), execution_len=execution_len,
        logical_len=logical_len, position=position, graph_mode=graph_mode)
    original = row.physical_descriptor
    stage_words = 15 + 4 * sum(original[11:14]) + 3 * original[14]
    if tuple(descriptor[2:stage_words]) != original[2:stage_words]:
        raise ValueError("native calibration changed the admitted chunk/span topology")
    metadata = {**dict(row.physical_metadata), **dict(decoded.physical_metadata)}
    if "layout_hash_hi" in metadata or "layout_hash_lo" in metadata:
        if not {"layout_hash_hi", "layout_hash_lo"}.issubset(metadata):
            raise ValueError("native calibration layout metadata is incomplete")
        metadata.update(layout_hash_hi=layout_hi, layout_hash_lo=layout_lo)
    return StageTuple.from_value(StageTuple(
        *decoded.as_tuple(), tuple(metadata.items()), decoded.physical_descriptor))


def plan_bounded_prefill_execution(
    real_len: int,
    physical_limit: int,
    padding_budget: int,
    *,
    resolve_stage_domain,
    position: int = 0,
    alignment: int = 1,
    padding_rows: str | int = "auto",
    exact_chunk_size: int | None = None,
    max_stage_chunk_size: int | None = None,
    candidate_logger=None,
    request_id=None,
    plan_result_sink=None,
    graph_mode="RETAINED_CACHE",
    queue_owner_id=0,
    lease_owner_id=0,
    physical_metadata=(),
    cost_scope=None,
    cost_certificates=(),
    execution_owner=None,
    execution_stage="prefill",
    execution_component=None,
    execution_native=None,
    plan_signature=None,
    graph_cache=None,
) -> tuple[int, int]:
    """Select once per prepared semantic signature, then reuse the native plan.

    Production callers explicitly declare resolver semantics in plan_signature.
    Ownerless diagnostic queries keep their existing uncached behavior.
    """

    if (graph_cache is not None and graph_cache.is_frozen()
            and (plan_signature is None or execution_owner is None)):
        raise RuntimeError("execution plan READY requires an explicit owner and semantic signature")

    cost_observer = None
    calibration = None
    if execution_owner is not None:
        if cost_scope is not None or cost_certificates:
            raise ValueError("production planning cannot override its owner-local cost binding")
        from rpu_backend.api._execution import planner_execution_context
        cost_scope, cost_certificates, cost_observer, calibration = planner_execution_context(
            execution_owner, execution_stage, execution_component, execution_native)
        if calibration is not None and not calibration.domain_scoped:
            if cost_scope is not None and cost_scope != calibration.scope:
                raise ValueError("calibration cannot override the installed owner scope")
            cost_scope = calibration.scope
    elif execution_native is not None:
        raise ValueError("native cost planning requires its actual execution owner")

    if candidate_logger is None and "HALO_CHUNK_DIAG" in os.environ:
        def candidate_logger(record):
            _LOG.info(
                "[A6_CHUNK_DIAG] %s",
                json.dumps(record, sort_keys=True, default=list),
            )

    physical_metadata = tuple(physical_metadata)
    def prepare():
        def typed_stage_domain(execution_len):
            try:
                return resolve_stage_domain(execution_len)
            except PlannerRejectError:
                raise
            except RuntimeError as exc:
                reject = native_chunk_reject(
                    exc,
                    stage="PREFILL",
                    requested=(
                        int(exact_chunk_size)
                        if exact_chunk_size is not None
                        else int(execution_len)
                    ),
                )
                if reject is None:
                    raise
                raise reject from exc

        return plan_prefill(
            real_len,
            physical_limit,
            padding_budget,
            position=position,
            alignment=alignment,
            padding_rows=padding_rows,
            exact_chunk_size=exact_chunk_size,
            max_stage_chunk_size=max_stage_chunk_size,
            logger=candidate_logger,
            request_id=request_id,
            graph_mode=graph_mode,
            queue_owner_id=queue_owner_id,
            lease_owner_id=lease_owner_id,
            physical_metadata=physical_metadata,
            cost_scope=cost_scope,
            cost_certificates=cost_certificates,
            cost_domain_observer=(
                (lambda length, rows: cost_observer("domain", (length, rows)))
                if cost_observer is not None else None),
            calibration_selection=calibration,
            calibration_mint=(
                (lambda length, row, route: _mint_planner_calibration_candidate(
                    execution_native, length, int(real_len), int(position), graph_mode, row, route))
                if calibration is not None and calibration.native_route is not None else None),
            resolve_stage_domain=typed_stage_domain,
        )

    if plan_signature is None or execution_owner is None:
        result = prepare()
    else:
        if execution_native is None:
            raise ValueError("cached production planning requires its actual native owner")
        if graph_cache is not None and (cost_observer is not None or calibration is not None):
            if graph_cache.is_frozen():
                raise RuntimeError("cost observation/calibration requires graph warmup, not READY execution")
        if cost_observer is not None or calibration is not None:
            # Collection must query the current native oracle, even when the
            # same descriptor already has a prepared production Graph.
            result = prepare()
        else:
            from rpu_backend.api._execution import prepared_execution_plan_cache

            prefix, handle = execution_native
            identity_op = getattr(torch.ops.rpu, f"{prefix}_planner_cache_identity", None)
            if identity_op is None:
                raise RuntimeError(f"{prefix}: native planner cache identity is unavailable; rebuild the backend")
            native_identity = tuple(identity_op(handle))
            if not native_identity:
                raise RuntimeError("native planner cache identity must not be empty")
            key = (
                execution_native, native_identity, execution_component, execution_stage,
                real_len, physical_limit, padding_budget, position, alignment,
                padding_rows, exact_chunk_size, max_stage_chunk_size,
                graph_mode, queue_owner_id, lease_owner_id, physical_metadata,
                plan_signature,
                # Installed owner costs are immutable. Native catalog changes
                # advance native_identity; Session installation clears local
                # preparation. Do not walk/hash the candidate-cost table here.
                None if cost_scope is None else (
                    cost_scope.profile_identity, cost_scope.dependency_identity,
                    cost_scope.runtime_identity),
            )
            if graph_cache is not None:
                result = graph_cache.prepare_plan(key, prepare)
            else:
                cache = prepared_execution_plan_cache(
                    execution_owner, execution_stage, execution_component, prefix)
                result = cache.prepare(key, prepare)
    if cost_observer is not None:
        cost_observer("result", (int(real_len), int(position), result))
    if plan_result_sink is not None:
        plan_result_sink(result)
    assert result.selected is not None
    return (
        int(result.selected.execution_len),
        int(result.selected.stage_tuple.compute_chunk),
    )


def plan_native_component_execution(
    logical_len: int,
    *,
    component_id: str,
    stage: str,
    generation: int,
    execution,
    resolve_stage_domain,
    graph_mode: str,
    physical_metadata=(),
    queue_owner_id: int = 0,
    request_id: str | None = None,
    cost_scope=None,
    cost_certificates=(),
    execution_owner=None,
    execution_component=None,
    execution_native=None,
    plan_signature=None,
    graph_cache=None,
    position: int = 0,
):
    """Plan one exact-length component through its native finite domain."""
    stage_config = execution.get(stage, {})
    exact = stage_config.get("chunk_size")
    box = {}
    execution_len, chunk = plan_bounded_prefill_execution(
        logical_len,
        logical_len,
        0,
        resolve_stage_domain=resolve_stage_domain,
        position=int(position),
        alignment=1,
        padding_rows=0,
        exact_chunk_size=exact if isinstance(exact, int) else None,
        graph_mode=graph_mode,
        cost_scope=cost_scope,
        cost_certificates=cost_certificates,
        execution_owner=execution_owner,
        execution_stage=stage,
        execution_component=execution_component,
        execution_native=execution_native,
        plan_signature=plan_signature,
        graph_cache=graph_cache,
        request_id=request_id or f"{component_id}:{stage}",
        physical_metadata=(
            ("execution_generation", int(generation)),
            *tuple(physical_metadata),
        ),
        queue_owner_id=int(queue_owner_id),
        plan_result_sink=lambda result: box.__setitem__("result", result),
    )
    result = box["result"]
    if (
        execution_len != logical_len
        or result.selected is None
        or not result.selected.stage_tuple.physical_descriptor
    ):
        raise RuntimeError(
            f"{component_id} planner returned no native descriptor"
        )
    return int(chunk), result


def _cold_text_cost_request(owner, native, cache, request, position, *,
                            component=None, stage="prefill"):
    """Validate a typed cold inspection without changing installed inputs."""
    from rpu_backend.api._execution import _planner_owner_binding, planner_cost_observer
    from rpu_backend.runtime.execution_planner import PlannerRequest

    session, component = _planner_owner_binding(owner, stage, component)
    if session is None or planner_cost_observer(owner, stage, component, native) is None:
        raise RuntimeError("text request inspection requires its cold cost collector")
    session.require_cold()
    if cache is not None and cache.position != 0:
        raise ValueError("cold text request inspection requires an empty actual KV cache")
    if type(position) is not int or position < 0:
        raise ValueError("cold text planning position must be a non-negative integer")
    if not isinstance(request, PlannerRequest):
        raise TypeError("cold text planning requires a typed PlannerRequest")
    fields = request.as_dict()
    if request.padding_rows is not None:
        if type(request.padding_budget) is not int or request.padding_budget != 0:
            raise ValueError("exact padding cannot carry a search budget")
        fields.pop("padding_budget")
    normalized = PlannerRequest.from_mapping(fields)
    if normalized != request or any(
        type(value) is not type(normalized.as_dict()[name])
        for name, value in request.as_dict().items()
    ):
        raise ValueError("cold text request must be canonically typed")
    known = session._planner_native.get((component, stage, native[0]))
    if known is not None and (known[0] is not owner or known[1] != native[1]):
        raise ValueError("cold text planning changed its actual native owner")
    return normalized, position


def _text_prefill_execution_plan(
    model, handle, past_key_values, seq_len: int, *, _cost_request=None, _cost_position=None
) -> tuple[int, int, object]:
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
    from rpu_backend.api._execution import (
        _PREFILL_PADDING_BUDGET_ENV, _PREFILL_PADDING_BUDGET_DEFAULT,
    )

    inspecting = _cost_request is not None or _cost_position is not None
    position = int(past_key_values.position)
    stage = getattr(model, "_rpu_execution", {}).get("prefill", {})
    planning_override = ()
    if inspecting:
        if type(seq_len) is not int or seq_len <= 1:
            raise ValueError("cold text prefill length must be an integer greater than one")
        if type(handle) is not int or handle != getattr(model, "_rpu_decoder_handle", None):
            raise ValueError("cold text planning requires its actual decoder handle")
        request, position = _cold_text_cost_request(
            model, ("causal_decoder", handle), past_key_values, _cost_request,
            position if _cost_position is None else _cost_position)
        stage = {"chunk_size": request.chunk_size or "auto",
                 "padding_rows": "auto" if request.padding_rows is None else request.padding_rows,
                 "padding_budget": request.padding_budget}
        planning_override = (request.chunk_size or 0,)
    padding_rows = stage.get("padding_rows", "auto")
    if isinstance(padding_rows, int) and not isinstance(padding_rows, bool):
        budget = 0  # exact rows do not consult the optional-search budget
    else:
        try:
            budget = int(stage.get(
                "padding_budget", _PREFILL_PADDING_BUDGET_DEFAULT
            ))
        except ValueError as exc:
            raise ValueError(
                f"{_PREFILL_PADDING_BUDGET_ENV} must be an integer") from exc
        if budget < 0:
            raise ValueError(
                f"{_PREFILL_PADDING_BUDGET_ENV} must be non-negative"
            )
    physical_limit = int(past_key_values.max_seq_len) - position
    plan_box = {}
    execution_len, chunk_size = plan_bounded_prefill_execution(
        seq_len, physical_limit, budget,
        execution_owner=model,
        execution_native=("causal_decoder", int(handle)),
        plan_signature=None if inspecting else (),
        graph_cache=getattr(model, "_rpu_decoder_graph_cache", None),
        resolve_stage_domain=lambda n: (
            torch.ops.rpu.causal_decoder_resolve_prefill_stage_domain(
                handle, n, position, True, 0, 0, 1, seq_len, *planning_override
            )
        ),
        position=position,
        alignment=1,
        padding_rows=padding_rows,
        exact_chunk_size=(
            stage.get("chunk_size")
            if isinstance(stage.get("chunk_size"), int)
            else None
        ),
        queue_owner_id=int(handle),
        plan_result_sink=lambda result: plan_box.__setitem__("result", result),
    )
    result = plan_box["result"]
    if not inspecting:
        vars(model)["_rpu_last_a6_plan"] = result.as_dict(
            include_candidates=False
        )
    return int(execution_len), int(chunk_size), result


def _text_decode_execution_plan(model, handle, *, position=0, graph_cache=None):
    """Observe the native fixed decode ABI without adding a decode config axis.

    Position zero preserves the shared mutable-position descriptor. A caller
    with scalar position registers selects an explicit position-specific domain.
    The shared preparation cache validates native identity and Graph READY;
    only explicit cost observation/calibration queries need a fresh oracle.
    """
    native = ("causal_decoder", int(handle))
    generation = int(getattr(model, "_rpu_execution_generation", 0))
    position = int(position)
    def domain(_length):
        # Position zero preserves the generic mutable-position ABI. An owner
        # using scalar position registers asks for a distinct COMPLETE domain.
        descriptor = tuple(torch.ops.rpu.causal_decoder_resolve_decode_stage_descriptor(
            handle, position) if position else
            torch.ops.rpu.causal_decoder_resolve_decode_stage_descriptor(handle))
        return [1, 1, len(descriptor), *descriptor]

    _, result = plan_native_component_execution(
        1, position=position, component_id="language_model", stage="decode", generation=generation,
        execution={}, resolve_stage_domain=domain, graph_mode=GRAPH_RETAINED_CACHE,
        queue_owner_id=int(handle), execution_owner=model, execution_native=native,
        plan_signature=(() if position == 0 else (position,)),
        graph_cache=(graph_cache if graph_cache is not None else
                     getattr(model, "_rpu_decoder_graph_cache", None)),
    )
    descriptor = result.selected.stage_tuple.physical_descriptor
    if position == 0:
        vars(model)["_rpu_decode_stage_descriptor"] = tuple(descriptor)
        # This diagnostic memo is not a second cache. Shared preparation above
        # owns all current native/cost/generation invalidation.
        key = (int(handle), generation, result.plan_digest, position)
        vars(model)["_decoder_decode_plan"] = (key, result)
    return result


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


# patch-reason: (b) Causal decoder forward runner — §3a (b)
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
    decode_descriptor=None,
    batch_slot=0,
    _validated_all_ones_attention_mask=None,
    **kwargs,
):
    """Shared RPU forward runner for the all-layers-once causal-decoder kernel.

    v5-11 Step-4 deviation rename: was `_rpu_qwen3_run_forward` at
    `_internal/patches/__init__.py:277`. The body is genuinely cross-arch
    (uses the generic `torch.ops.rpu.causal_decoder_forward` op family
    renamed v5-06 D-06+D-07); the "qwen3" name was historical misnomer.
    The shared runtime owns the architecture-independent forward path.

    Returns:
        (raw_tensor, past_key_values)
        raw_tensor is [batch, seq_len, hidden_size] hidden states.
        (Phase 2.5: C++ post_graph for shape-changing logits output is deferred;
        the C++ side always returns hidden states.)
    """
    # Late import (avoids module-load circular with api.cache).
    from rpu_backend.api.cache import RPUCache
    # D-32 wire-point #3 (first-forward marker). Idempotent per D-22 — safe
    # to call every forward.
    from rpu_backend.runtime.hw_attrs import mark_first_forward_done
    mark_first_forward_done(model)

    # HF contract: input_ids XOR inputs_embeds
    if (input_ids is None) == (inputs_embeds is None):
        raise AssertionError(
            "RPU all-layers-once: must provide exactly one of "
            "input_ids or inputs_embeds (HF contract modeling_qwen3.py:401)"
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
    validate_decoder_cache_topology(getattr(model, "_rpu_decoder_topology", None),
        past_key_values, batch_size=int((inputs_embeds if inputs_embeds is not None else input_ids).shape[0]))

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
            "(Phase 1 only returns OutputWithPast)"
        )
    if use_cache is False:
        raise AssertionError(
            "RPU all-layers-once: use_cache=False is not supported "
            "in Phase 1 (caching path only; no-cache training scenario deferred)"
        )
    # cache_position is received but ignored (RPUCache.position owns position).
    _ = cache_position
    # R-Phase 1 (Qwen3-VL): position_ids is forwarded to the C++ op when
    # M-RoPE is active. Non-mrope models (Qwen3 / Llama) pass position_ids=None
    # to the op — the C++ side silently ignores it.
    #
    # HF's Qwen3-VL forward emits position_ids in shape [3, batch, seq_len];
    # the C++ M-RoPE entry check expects [seq_len, 3] int32 RPU contig.
    # Normalize here so adapters don't each duplicate the reshape.
    if position_ids is not None and position_ids.dim() == 3:
        # [3, batch, seq_len] → [seq_len, 3]; only batch==1 supported in R-Phase 1.
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
    logical_seq_len = seq_len
    planned_chunk_size = 0
    planned_stage_descriptor = ()
    # Native decode descriptors are resolved with RETAINED_CACHE at install;
    # prefill carries the lifecycle of the A6 winner consumed below.
    planned_graph_mode = GRAPH_RETAINED_CACHE
    if prefill_plan is not None:
        execution_len, planned_chunk_size, plan_result = prefill_plan
        execution_len, planned_chunk_size = map(
            int, (execution_len, planned_chunk_size)
        )
        assert plan_result.selected is not None
        planned_stage_descriptor = (
            plan_result.selected.stage_tuple.physical_descriptor
        )
        if not planned_stage_descriptor:
            raise RuntimeError("A6 prefill winner has no native stage descriptor")
        planned_graph_mode = plan_result.graph_mode
    elif (seq_len > 1
          and position_ids is None
          and (attention_mask is None
               or _validated_all_ones_attention_mask
               is _BATCH_PREFILL_ALL_ONES_MASK)):
        prefill_plan = _text_prefill_execution_plan(
            model, handle, past_key_values, seq_len)
        execution_len, planned_chunk_size, plan_result = prefill_plan
        planned_stage_descriptor = (
            plan_result.selected.stage_tuple.physical_descriptor
        )
        planned_graph_mode = plan_result.graph_mode

    if decode_descriptor is not None:
        if seq_len != 1 or prefill_plan is not None:
            raise RuntimeError("an explicit decode descriptor requires a single-token decode")
        planned_stage_descriptor = tuple(decode_descriptor)
        if not planned_stage_descriptor:
            raise RuntimeError("an explicit decode descriptor cannot be empty")

    if seq_len == 1 and not planned_stage_descriptor:
        planned_stage_descriptor = tuple(
            getattr(model, "_rpu_decode_stage_descriptor", ())
        )
        if not planned_stage_descriptor:
            raise RuntimeError(
                "causal decoder decode has no installed physical descriptor"
            )

    if planned_chunk_size:
        if seq_len <= 1:
            raise RuntimeError(
                "an A6 prefill plan cannot be applied to decode"
            )
        if execution_len < seq_len:
            raise RuntimeError(
                f"prefill execution_len={execution_len} is smaller than "
                f"logical seq_len={seq_len}"
            )
        # Plain text arrives at its logical length; M-RoPE composites pad
        # embeddings, positions and visual rows before entering this runner.
        # Both carry the same A6 winner, whose padding records the real token
        # count. Tensor shape alone cannot determine cache/receipt semantics.
        selected = plan_result.selected
        logical_seq_len = int(selected.execution_len) - int(selected.padding_rows)
        if (
            int(selected.execution_len) != execution_len
            or int(selected.stage_tuple.compute_chunk) != planned_chunk_size
            or not 1 < logical_seq_len <= seq_len <= execution_len
            or seq_len not in (logical_seq_len, execution_len)
        ):
            raise RuntimeError(
                "causal prefill input/physical plan mismatch: "
                f"input_len={seq_len}, logical_len={logical_seq_len}, "
                f"execution_len={execution_len}, planned_chunk={planned_chunk_size}, "
                f"selected_execution_len={selected.execution_len}, "
                f"selected_chunk={selected.stage_tuple.compute_chunk}"
            )
        if execution_len > seq_len:
            if position_ids is not None or (
                attention_mask is not None
                and _validated_all_ones_attention_mask
                is not _BATCH_PREFILL_ALL_ONES_MASK
            ):
                raise RuntimeError(
                    "externally planned M-RoPE/masked prefill must pad all "
                    "semantic inputs before the shared decoder call"
                )
            hidden_states = torch.cat(
                [hidden_states,
                 hidden_states.new_zeros(hidden_states.shape[0],
                                         execution_len - seq_len,
                                         hidden_states.shape[2])],
                dim=1,
            ).contiguous()

    # R-Phase 2 (Qwen3-VL): deepstack_dense_visual_embeds is forwarded to the
    # C++ op when DeepStack is active. None default → empty list (transparent
    # for Qwen3 / Llama 1D-RoPE callers). Qwen3-VL callers pass either
    # zero-keepalive views (text-only / decode) or scattered dense embeds
    # (image prefill).
    if deepstack_dense_visual_embeds is None:
        deepstack_dense_visual_embeds = []

    # Packed KV tensors round storage to 16 rows. Enforce the public logical
    # limit before native execution, including temporary prefill padding, so an
    # overflowing decode cannot write an aligned tail before update_position().
    if (past_key_values.position < 0 or
            past_key_values.position + execution_len > past_key_values.max_seq_len):
        raise ValueError(
            "RPUCache position overflow before execution: "
            f"position={past_key_values.position}, execution_len={execution_len}, "
            f"max_seq_len={past_key_values.max_seq_len}"
        )

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
        position_ids,                   # R-Phase 1: forwarded to mrope path; None otherwise
        deepstack_dense_visual_embeds,  # R-Phase 2: DeepStack injection (or [])
        rope_cos_il,                    # partial_mrope: host-baked interleaved cos (None → legacy
        rope_sin_il,                    #   in-kernel llama_mrope_interleave; both None = unchanged)
        -1,                             # cos_sin_offset: -1 = legacy ctx().position base
        batch_slot,                     # batch decode: KV-cache slot for this batch==1 call
        batch_decode_enabled,           # explicit per-model capability; Qwen3 only
        0,                              # descriptor is the sole plan authority
        planned_stage_descriptor,       # complete A6 physical winner
    )

    if planned_chunk_size:
        # The successful native call consumed this selected descriptor. FMB
        # validates its compute/qkv capacities and schedules against the live
        # request; querying its saved copy of that capacity adds no evidence.
        resolved_chunk_size = int(plan_result.selected.stage_tuple.compute_chunk)
    else:
        resolved_chunk_size = int(
            torch.ops.rpu.causal_decoder_get_resolved_chunk_size(handle)
        )

    vars(model)["_rpu_last_execution_plan"] = {
        "stage": "prefill" if logical_seq_len > 1 else "decode",
        "component": getattr(
            model, "_rpu_execution_component_id", "language_model"
        ),
        "generation": int(getattr(model, "_rpu_execution_generation", 0)),
        "logical_len": int(logical_seq_len),
        "execution_len": int(execution_len),
        "chunk_size": resolved_chunk_size,
        "padding_rows": int(execution_len - logical_seq_len),
        "position": int(past_key_values.position),
        "physical_descriptor": tuple(planned_stage_descriptor),
        "graph_mode": planned_graph_mode,
    }

    # Slice the padded tail off before anyone sees it, and advance the cache by
    # the LOGICAL length so decode continues from the real end (and overwrites
    # the first padded KV row).
    if execution_len > logical_seq_len:
        raw = raw[:, :logical_seq_len]

    # Global position advance (the fused kernel does not touch RPUCache state)
    past_key_values.update_position(logical_seq_len)

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


# Phase 06.1 / D-6-17 (codex round-1 + round-2): generic CausalLM all-layers-once
# fusion seam. Body shared between Qwen3 + Llama (and future Phi/Mistral).
# Qwen3-specific q_norm/k_norm reads stay on the Qwen3 surface, behind the
# `qk_norm_lists=` kwarg gate.
#
# v5-04 HND-01: legacy per-arch handle alias removed (hard cutover per
# ADR §10 #1). All callers now read `_rpu_decoder_handle` directly.
# v5-11 Step-4 deviation: renamed from `patch_causal_decoder_for_rpu` to
# `_install_causal_decoder_forward`; co-located with `_run_causal_decoder_forward`
# (was `_rpu_qwen3_run_forward`) to satisfy ADR §3.2 DAG (runtime ↛ adapters).
def build_mrope_cos_sin_tables(
    text_config: Any,
    *,
    max_seq_len: int | None = None,
    device: str | torch.device = "rpu",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build kernel-format M-RoPE cos/sin tables.

    Shape: `[max_seq_len, head_dim / 2]` (the rotary half-dim — the kernel
    consumes the standard 1D RoPE table; the per-axis (T/H/W) selection is
    performed at runtime via strobe masks set by the launcher).

    Args:
        text_config: Qwen3VLTextConfig (or anything quack-compatible with
            head_dim / num_attention_heads / hidden_size / rope_parameters).
        max_seq_len: Defaults to `text_config.max_position_embeddings`.
            R-Phase 1 sizes the table to match the runtime keepalive
            (QWEN3_MROPE_MAX_KEEPALIVE_SEQ = 8192). Larger values are safe
            but waste DDR.
        device: target device (default "rpu").
        dtype: target dtype (default torch.float16 — matches the kernel
            register format).
    """
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None:
        head_dim = text_config.hidden_size // text_config.num_attention_heads
    if head_dim <= 0 or (head_dim % 2) != 0:
        raise ValueError(
            f"build_mrope_cos_sin_tables: head_dim must be positive and even, "
            f"got {head_dim}"
        )

    # rope_parameters (transformers >=5.x) replaces rope_scaling. Try both for
    # forward-compat with mid-flight transformers versions.
    rope_params = getattr(text_config, "rope_parameters", None)
    if rope_params is None:
        rope_params = getattr(text_config, "rope_scaling", None) or {}
    rope_type = rope_params.get("rope_type", "default")
    if rope_type != "default":
        # YaRN / dynamic / linear scaling cases need the corresponding HF
        # rope_init_fn. R-Phase 1 only wires the default branch — bail loudly
        # rather than silently produce wrong cos/sin.
        raise NotImplementedError(
            f"build_mrope_cos_sin_tables: rope_type={rope_type!r} not yet "
            "supported; only 'default' for Qwen3-VL 2B/4B-Instruct is wired."
        )
    rope_theta = rope_params.get("rope_theta", None)
    if rope_theta is None:
        rope_theta = getattr(text_config, "rope_theta", 10000.0)

    if max_seq_len is None:
        max_seq_len = int(getattr(text_config, "max_position_embeddings", 8192))

    half = head_dim // 2
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    )  # [half]
    positions = torch.arange(max_seq_len, dtype=torch.float64)  # [max_seq]
    freqs = positions[:, None] * inv_freq[None, :]  # [max_seq, half]
    cos = freqs.cos().to(dtype=dtype)
    sin = freqs.sin().to(dtype=dtype)
    cos = cos.to(device=device).contiguous()
    sin = sin.to(device=device).contiguous()
    expected_shape = (max_seq_len, half)
    if tuple(cos.shape) != expected_shape or tuple(sin.shape) != expected_shape:
        raise RuntimeError(
            "build_mrope_cos_sin_tables produced invalid table shapes: "
            f"cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, "
            f"expected={expected_shape}"
        )
    return cos, sin


def validate_qwen3_vl_text_semantics(text_model) -> None:
    """Check live scalar/module semantics before admitting reduced VL text."""
    from numbers import Integral
    from rpu_backend.api._execution import qwen3_vl_text_core_profile
    from transformers.activations import SiLUActivation

    profile = qwen3_vl_text_core_profile(text_model.config, 4)
    for index, layer in enumerate(text_model.layers):
        activation = getattr(layer.mlp, "act_fn", None)
        # Transformers5 uses its exact nn.functional.silu wrapper; earlier
        # versions use nn.SiLU. Both have the same admitted activation math.
        if not (type(activation) is SiLUActivation or (
                type(activation) is torch.nn.SiLU and activation.inplace is False)):
            raise ValueError(f"reduced-core Qwen3-VL layer {index} requires exact SiLU activation")
        for owner, name in ((layer.self_attn, "q_norm"), (layer.self_attn, "k_norm"),
                            (layer, "input_layernorm"), (layer, "post_attention_layernorm")):
            if getattr(getattr(owner, name, None), "variance_epsilon", None) != 1e-6:
                raise ValueError(f"reduced-core Qwen3-VL layer {index} {name} requires epsilon 1e-6")
    if getattr(text_model.norm, "variance_epsilon", None) != 1e-6:
        raise ValueError("reduced-core Qwen3-VL final norm requires epsilon 1e-6")

    rotary = getattr(text_model, "rotary_emb", None)
    section = getattr(rotary, "mrope_section", None)
    if (getattr(rotary, "rope_type", None) != "default"
            or getattr(rotary, "attention_scaling", None) != 1.0
            or not isinstance(section, (list, tuple))
            or tuple(section) != (24, 20, 20)
            or any(isinstance(item, bool) or not isinstance(item, Integral) for item in section)
            or qwen3_vl_text_core_profile(getattr(rotary, "config", None), 4) != profile):
        raise ValueError("reduced-core Qwen3-VL requires exact live default interleaved M-RoPE semantics")


def _validate_decoder_w8_scale_lists(model, scale_lists):
    """Bind all seven INT8 projections to their actual per-channel buffers."""
    if (scale_lists is None or len(scale_lists) != 7
            or any(len(scales) != len(model.layers) for scales in scale_lists)):
        raise ValueError("reduced-core W8A16 requires seven complete scale lists")
    roles = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    for role, scales in zip(roles, scale_lists, strict=True):
        for index, (layer, scale) in enumerate(zip(model.layers, scales, strict=True)):
            projection = layer.get_submodule(role)
            if (projection.weight.dtype != torch.int8
                    or not isinstance(scale, torch.Tensor)
                    or scale is not getattr(projection, "weight_scale", None)
                    or scale.dtype != torch.float16
                    or tuple(scale.shape) != (projection.out_features,)
                    or not scale.is_contiguous()
                    or scale.device != projection.weight.device):
                raise ValueError(f"reduced-core W8A16 layer {index} {role} requires its actual INT8 weight / FP16 per-channel scale")


def _validate_decoder_w4_scale_lists(model, scale_lists, topology):
    """Check the exact dense/G32/TP8 packed install before native creation."""
    from rpu_backend.quant.qwen3_profiles import qwen3_dense_quant_profile

    profile = qwen3_dense_quant_profile(model.config)
    if topology.num_cores != 8 or len(model.layers) != profile.num_layers:
        raise ValueError("packed Qwen3 topology requires the exact dense/G32/TP8 recipe")
    if (scale_lists is None or len(scale_lists) != 7
            or any(len(scales) != len(model.layers) for scales in scale_lists)):
        raise ValueError("packed Qwen3 requires seven complete G32 scale lists")
    h, i = profile.hidden_size, profile.intermediate_size
    q, kv = profile.num_q_heads * profile.head_dim, profile.num_kv_heads * profile.head_dim
    roles = (("self_attn.q_proj", q, h, 1),
             ("self_attn.k_proj", kv, h, 1),
             ("self_attn.v_proj", kv, h, 1),
             ("self_attn.o_proj", h, q, 0),
             ("mlp.gate_proj", i, h, 1),
             ("mlp.up_proj", i, h, 1),
             ("mlp.down_proj", h, i, 0))
    for (role, n, k, partition), scales in zip(roles, scale_lists, strict=True):
        local_g, local_n = (k // 32, n // 8) if partition else (k // 32 // 8, n)
        physical_scales = 8 * ((local_g + 3) // 4 * 4) * ((local_n + 63) // 64 * 64)
        for index, (layer, scale) in enumerate(zip(model.layers, scales, strict=True)):
            projection = layer.get_submodule(role)
            weight = projection.weight
            if ((projection.out_features, projection.in_features) != (n, k)
                    or weight.dtype != torch.uint8 or tuple(weight.shape) != (n, k // 2)
                    or not weight.is_contiguous()
                    or not isinstance(scale, torch.Tensor)
                    or scale is not getattr(projection, "weight_scale", None)
                    or scale.dtype != torch.float16 or not scale.is_contiguous()
                    or scale.device != weight.device
                    or tuple(scale.shape) != (32, physical_scales // 32)):
                raise ValueError(f"packed Qwen3 layer {index} {role} requires its actual UINT8/G32 payload")


def _validate_mrope_decoder_topology(
    model, *, config, arch, execution_config, topology, mrope_section,
    deepstack_lang_layers, scale_lists,
):
    """Admit the bounded M-RoPE profile before allocating wrapper resources."""
    if execution_core_count(execution_config) != (topology.num_cores if topology else 8):
        raise ValueError("decoder install requires the resolved cold model topology")
    if topology is None or topology.num_cores == 8:
        return
    from numbers import Integral
    from rpu_backend.api._execution import qwen3_vl_text_core_profile

    if arch != "qwen3_vl_text":
        raise ValueError("reduced-core M-RoPE requires the exact Qwen3-VL text profile")
    profile = qwen3_vl_text_core_profile(config, topology.num_cores)
    bound_profile = qwen3_vl_text_core_profile(model.config, topology.num_cores)
    w8a16 = scale_lists is not None
    if (profile != bound_profile or len(model.layers) != profile.num_layers
            or (w8a16 and profile.hidden_size != 2048)):
        raise ValueError("reduced-core M-RoPE requires exact FP16 text or 2B per-channel W8A16 layers")
    if w8a16:
        _validate_decoder_w8_scale_lists(model, scale_lists)
    if resolve_decoder_topology(num_cores=topology.num_cores, **profile.geometry()) != topology:
        raise ValueError("decoder geometry differs from the bound cold topology")
    validate_qwen3_vl_text_semantics(model)
    for value, expected, label in (
        (mrope_section, (24, 20, 20), "M-RoPE sections"),
        (deepstack_lang_layers, (0, 1, 2), "DeepStack consumers"),
    ):
        if (value is None or tuple(value) != expected
                or any(isinstance(item, bool) or not isinstance(item, Integral)
                       for item in value)):
            raise ValueError(f"reduced-core Qwen3-VL requires {label} {list(expected)}")
    hidden = profile.hidden_size
    intermediate = decoder_mlp_intermediate_size(profile.intermediate_size, topology.mlp_tp)
    shapes = {
        "q_proj": (profile.num_q_heads * profile.head_dim, hidden),
        "k_proj": (profile.num_kv_heads * profile.head_dim, hidden),
        "v_proj": (profile.num_kv_heads * profile.head_dim, hidden),
        "o_proj": (hidden, profile.num_q_heads * profile.head_dim),
        "gate_proj": (intermediate, hidden), "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }
    projection_dtype = torch.int8 if w8a16 else torch.float16
    for index, layer in enumerate(model.layers):
        for owner, names in (
            (layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
            (layer.mlp, ("gate_proj", "up_proj", "down_proj")),
        ):
            for name in names:
                projection = getattr(owner, name)
                if (projection.weight.dtype != projection_dtype
                        or tuple(projection.weight.shape) != shapes[name]
                        or (projection.out_features, projection.in_features) != shapes[name]
                        or getattr(projection, "bias", None) is not None):
                    raise ValueError(f"reduced-core Qwen3-VL layer {index} {name} requires exact {projection_dtype} weights")
        for norm, width in ((layer.self_attn.q_norm, profile.head_dim),
                            (layer.self_attn.k_norm, profile.head_dim),
                            (layer.input_layernorm, hidden),
                            (layer.post_attention_layernorm, hidden)):
            if norm.weight.dtype != torch.float16 or tuple(norm.weight.shape) != (width,):
                raise ValueError(f"reduced-core Qwen3-VL layer {index} requires exact FP16 norms")
    if model.norm.weight.dtype != torch.float16 or tuple(model.norm.weight.shape) != (hidden,):
        raise ValueError("reduced-core Qwen3-VL requires an exact FP16 final norm")


def install_mrope_text_decoder_for_rpu(
    text_model,
    *,
    arch: str,
    chunk_envelope_for,
    text_config: Any | None = None,
    max_seq_len: int | None = None,
    vision_config: Any | None = None,
    deepstack_lang_layers: list[int] | None = None,
    enable_deepstack: bool = True,
    scale_lists=None,
    execution_config=None,
    topology=None,
    _kv_cache_layer_bank_size=1,
    _graph_cache_max_entries=None,
) -> int:
    """Install RPU all-layers-once forward on a `Qwen3VLTextModel` instance.

    Prerequisites (same as the Qwen3 path):
      - `text_model.to('rpu')` has been called.
      - `convert_linear_weights_inplace(text_model)` has been called.

    Text path with M-RoPE and optional DeepStack injection wired through
    `causal_decoder_set_weights`. The supported Qwen3-VL ConditionalGeneration
    adapter composes this with the Vision tower; component/text-only callers can
    opt out of DeepStack via `enable_deepstack=False` or an empty layer list.

    Args:
        text_model: Qwen3VLTextModel instance (already moved to RPU).
        text_config: optional; defaults to `text_model.config`.
        max_seq_len: cos/sin table length; defaults to
            text_config.max_position_embeddings.
        vision_config: optional; used only to infer the number of leading text
            layers that receive DeepStack when `deepstack_lang_layers` is None.
            Pass `model.config.vision_config` for end-to-end Qwen3-VL flows.
        deepstack_lang_layers: explicit list of text-decoder layer indices that
            receive DeepStack injection. Overrides vision_config detection.
        enable_deepstack: false → bypass DeepStack injection regardless of
            config (text-only smoke / R-Phase 1 fallback).
        scale_lists: optional W8A16 per-output-channel fp16 scale tuple
            (q,k,v,o,gate,up,down), one fp16 [out] tensor per layer per
            projection. When provided, the decoder's 7 projection weights must
            already be int8 (swizzled dwidth=1) and the install routes to
            `causal_decoder_set_weights_w8a16`. None → fp16 (default).

    Returns the C++ handle (also stashed at `text_model._rpu_decoder_handle`).
    """
    import rpu_backend

    if type(_kv_cache_layer_bank_size) is not int or _kv_cache_layer_bank_size < 1:
        raise ValueError("M-RoPE KV cache layer bank size must be a positive integer")
    if _graph_cache_max_entries is not None and (
        type(_graph_cache_max_entries) is not int or _graph_cache_max_entries < 1
    ):
        raise ValueError("_graph_cache_max_entries must be a positive integer or None")

    old_resource = getattr(text_model, "_rpu_decoder_retirement_state", None)
    if old_resource is not None:
        old_resource.require_replaceable()
    else:
        from rpu_backend.api._execution import _require_execution_process_safe
        _require_execution_process_safe()

    cfg = text_config if text_config is not None else getattr(text_model, "config", None)
    if cfg is None:
        raise ValueError(
            "install_qwen3_vl_text_for_rpu: text_config not provided and "
            "text_model has no .config attribute."
        )

    rope_params = getattr(cfg, "rope_parameters", None)
    if rope_params is None:
        rope_params = getattr(cfg, "rope_scaling", None) or {}
    mrope_section = rope_params.get("mrope_section", None)
    if mrope_section is None:
        raise ValueError(
            "install_qwen3_vl_text_for_rpu: text_config.rope_parameters['mrope_section'] "
            "is required (expected [T_dim_half, H_dim_half, W_dim_half])."
        )
    # Vision indexes select extraction blocks; HF injects those outputs into
    # the first text layers in list order.  Caller > vision_config > [].
    if not enable_deepstack:
        ds_layers: list[int] = []
    elif deepstack_lang_layers is not None:
        ds_layers = list(deepstack_lang_layers)
    elif vision_config is not None:
        ds_layers = list(
            range(len(getattr(vision_config, "deepstack_visual_indexes", [])))
        )
    else:
        ds_layers = []

    if (hasattr(text_model, "_rpu_decoder_handle")
            and getattr(text_model, "_rpu_decoder_topology", None) != topology):
        raise ValueError("decoder replacement cannot change the cold weight topology")
    _validate_mrope_decoder_topology(
        text_model, config=cfg, arch=arch, execution_config=execution_config,
        topology=topology, mrope_section=mrope_section,
        deepstack_lang_layers=ds_layers, scale_lists=scale_lists,
    )
    mrope_section = [int(x) for x in mrope_section]
    ds_layers = [int(x) for x in ds_layers]
    cos, sin = build_mrope_cos_sin_tables(
        cfg, max_seq_len=max_seq_len, device="rpu", dtype=torch.float16,
    )

    # Capture the text decoder under its cold topology and cache capacity.
    text_graph_cache = _new_decoder_graph_cache(
        topology, max_entries=_graph_cache_max_entries)
    text_num_layers = int(cfg.num_hidden_layers)
    text_hidden_size = int(cfg.hidden_size)
    # FNV1a-style hash of the deepstack lang-layer list — used as a
    # tiebreaker in the GraphSignature so two models with different
    # deepstack layouts but the same sequence shape don't alias.
    _ds_hash = 0
    for idx in ds_layers:
        _ds_hash = (_ds_hash * 1099511628211) ^ int(idx)
        _ds_hash &= (1 << 63) - 1
    runtime_attrs = {
        "_rpu_kv_cache_layer_bank_size": _kv_cache_layer_bank_size,
        "_rpu_text_graph_cache": text_graph_cache,
        "_rpu_text_num_layers": text_num_layers,
        "_rpu_text_hidden_size": text_hidden_size,
        "_rpu_text_deepstack_hash": _ds_hash,
    }
    model_vars = vars(text_model)
    snapshot = {
        name: (name in model_vars, model_vars.get(name))
        for name in runtime_attrs
    }

    # Publish only pre-built, non-native wrapper state before entering the
    # common decoder transaction. If that transaction fails, its old handle
    # remains published and this wrapper state is restored exactly.
    try:
        for name, value in runtime_attrs.items():
            setattr(text_model, name, value)
        handle = _install_causal_decoder_forward(
            text_model,
            arch=arch,
            mrope_section=mrope_section,
            cos_sin=(cos, sin),
            deepstack_lang_layers=ds_layers,
            scale_lists=scale_lists,
            chunk_envelope_for=chunk_envelope_for,
            execution_config=execution_config,
            topology=topology,
            **({"_graph_cache_max_entries": _graph_cache_max_entries}
               if _graph_cache_max_entries is not None else {}),
        )
    except BaseException:
        # Rollback must not re-enter model-defined attribute hooks: the
        # publication failure may have come from those hooks in the first
        # place, and they may keep rejecting subsequent writes/deletes.
        model_vars = vars(text_model)
        for name in runtime_attrs:
            model_vars.pop(name, None)
        for name, (had_attr, value) in snapshot.items():
            if had_attr:
                model_vars[name] = value
        raise

    return handle


def pad_mrope_prefill_inputs(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    execution_len: int,
):
    """Right-pad token-indexed inputs to the selected execution length.

    Padded rows cannot affect the logical prefix under the causal
    lower-triangular mask (LTM).
    """
    logical_len = inputs_embeds.size(1)
    pad = execution_len - logical_len
    if pad < 0:
        raise ValueError(
            f"execution_len={execution_len} is smaller than logical_len={logical_len}"
        )
    if pad == 0:
        return inputs_embeds, attention_mask, position_ids

    inputs_embeds = torch.cat(
        [inputs_embeds,
         inputs_embeds.new_zeros(inputs_embeds.size(0), pad, inputs_embeds.size(2))],
        dim=1,
    )
    if attention_mask is not None and attention_mask.dim() == 2:
        if attention_mask.size(-1) != logical_len:
            raise ValueError(
                "M-RoPE decoder forward: 2D attention_mask length "
                f"{attention_mask.size(-1)} != input length {logical_len}"
            )
        attention_mask = torch.cat(
            [attention_mask,
             attention_mask.new_zeros(attention_mask.size(0), pad)],
            dim=-1,
        )
    if position_ids.dim() == 3 and position_ids.size(-1) == logical_len:
        position_ids = torch.cat(
            [position_ids,
             position_ids[..., -1:].expand(*position_ids.shape[:-1], pad)],
            dim=-1,
        ).contiguous()
    elif (
        position_ids.dim() == 2
        and position_ids.size(0) == logical_len
        and position_ids.size(1) == 3
    ):
        position_ids = torch.cat(
            [position_ids, position_ids[-1:].expand(pad, 3)],
            dim=0,
        ).contiguous()
    else:
        raise ValueError(
            "M-RoPE decoder forward: position_ids must be [3, batch, seq] "
            f"or [seq, 3] with seq={logical_len}, got "
            f"{tuple(position_ids.shape)}"
        )
    return inputs_embeds, attention_mask, position_ids


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
    topology=None,
    _graph_cache_max_entries=None,
) -> int:
    """Generic CausalLM all-layers-once installer (D-06: single shared body for Qwen3 + Llama).

    Phase 06.1 / D-6-17 round-2 (codex HIGH-1+HIGH-2): the C++ kernel side
    accepts empty per-layer q_norm/k_norm tensor lists post-T1.5 and routes on
    an explicit `has_qk_norm_` boolean. Qwen3 callers pass the gathered q/k norm
    lists; Llama callers pass `qk_norm_lists=None` (which materializes as empty
    lists at the C++ boundary).

    **Transactional + idempotent**: fully configure the replacement before
    retiring the old install. Successful rollback preserves the old state;
    uncertain graph/native retirement retains both installs and fails closed.

    Prerequisites:
      - model.to("rpu") must have been called (weights on RPU device)
      - convert_linear_weights_inplace(model) must have been called
        (applies row/col partition swizzle to Linear weights)

    Args:
        model: A decoder model instance (Qwen3Model or LlamaModel — the inner base, not
            the outer ForCausalLM wrapper).
        arch: "qwen3" — Qwen3-specific q_norm/k_norm gathered from
              model.layers[i].self_attn.q_norm/k_norm.
              "llama" — empty q_norm/k_norm tensor lists passed to the C++ kernel
              (has_qk_norm_=false branch; in-place RoPE per Phase 06.1 D-6-17 round-2).
              "qwen3_vl_text" — Qwen3-VL text decoder; same QK-norm layout as Qwen3,
              additionally requires `mrope_section` and kernel-format `cos_sin`.
        qk_norm_lists: Internal escape hatch — if provided (Qwen3 wrapper passes the
            tuple it already gathered), short-circuits arch-specific gather. New callers
            should pass `arch=` only.
        mrope_section: R-Phase 1 (Qwen3-VL). Required for arch='qwen3_vl_text';
            rejected for non-mrope archs. Forwarded as the trailing append-only
            kwarg of `causal_decoder_set_weights`.
        cos_sin: Override for the kernel-format cos/sin tables `[max_seq, head_dim/2]`.
            Required for mrope (kernel rejects HF-expanded `[max_seq, head_dim]`).
            None → fall back to rotary_emb introspection (legacy 1D-RoPE).

    Returns:
        int: The handle for this model. Also stored as `model._rpu_decoder_handle`.
    """
    if _graph_cache_max_entries is not None and (
        type(_graph_cache_max_entries) is not int or _graph_cache_max_entries < 1
    ):
        raise ValueError("_graph_cache_max_entries must be a positive integer or None")
    # R-Phase 1 (Qwen3-VL GAP-A1): 'qwen3_vl_text' is the M-RoPE-enabled variant.
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

    if execution_config is None and arch != "qwen3_vl_text":
        execution_config = getattr(model, "_rpu_execution", None)
    execution_config = dict(execution_config or {})
    if execution_core_count(execution_config) != (topology.num_cores if topology else 8):
        raise ValueError("decoder install requires the resolved cold model topology")
    if topology is not None and arch not in ("qwen3", "qwen3_vl_text"):
        raise ValueError("reduced-core decoder topology is admitted only by Qwen3 and Qwen3-VL")
    if topology is not None and topology.num_cores != 8:
        if arch == "qwen3_vl_text":
            _validate_mrope_decoder_topology(
                model, config=model.config, arch=arch, execution_config=execution_config,
                topology=topology, mrope_section=mrope_section,
                deepstack_lang_layers=deepstack_lang_layers, scale_lists=scale_lists,
            )
        else:
            from rpu_backend.api._execution import qwen3_core_profile, is_qwen3_17b_w8a16_core_config
            profile = qwen3_core_profile(model.config, topology.num_cores)
            w8a16 = is_qwen3_17b_w8a16_core_config(model.config)
            if len(model.layers) != profile.num_layers or (scale_lists is not None) != w8a16:
                raise ValueError(f"reduced-core decoder install requires all {profile.num_layers} admitted FP16 or W8A16 layers")
            if w8a16:
                _validate_decoder_w8_scale_lists(model, scale_lists)
            if resolve_decoder_topology(num_cores=topology.num_cores,
                                        **profile.geometry()) != topology:
                raise ValueError("decoder geometry differs from the bound cold topology")
    if arch != "qwen3_vl_text" and getattr(model, "_execution_session", None) is None:
        from rpu_backend.api._execution import _cold_causal_decoder_execution

        # Direct low-level installs have no outer Session; keep their cold
        # translation too. Bound owners already carry the authoritative view.
        execution_config = _cold_causal_decoder_execution(execution_config)

    # W-05: validate `arch` against model.config.model_type to catch caller-side
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
    if old_handle is not _missing and getattr(model, "_rpu_decoder_topology", None) != topology:
        raise ValueError("decoder replacement cannot change the cold weight topology")
    old_finalizer = getattr(model, "_rpu_decoder_handle_finalizer", None)
    old_resource = getattr(model, "_rpu_decoder_retirement_state", None)
    if old_resource is not None:
        old_resource.require_replaceable()
    else:
        from rpu_backend.api._execution import _require_execution_process_safe
        _require_execution_process_safe()
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
        q_norm_list, k_norm_list = [], []  # empty per-layer norm — kernel accepts post-T1.5
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
    if topology is not None:
        if arch == "qwen3" and any(getattr(weight, "dtype", None) == torch.uint8
                for group in (q_w_list, k_w_list, v_w_list, o_w_list, gate_list, up_list, down_list)
                for weight in group):
            _validate_decoder_w4_scale_lists(model, scale_lists, topology)
        # Model config remains logical after one-time CPU weight padding.
        # Native admission takes logical geometry, while weight/SPM execution
        # uses the physical width selected by the same cold template.
        intermediate_size = model.config.intermediate_size
        physical_intermediate = decoder_mlp_intermediate_size(intermediate_size, topology.mlp_tp)
        actual = resolve_decoder_topology(
            num_cores=topology.num_cores, hidden_size=hidden_size,
            intermediate_size=intermediate_size, num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            vocab_size=model.config.vocab_size)
        if actual != topology:
            raise ValueError("decoder layer geometry does not match the bound topology")
        for index, layer in enumerate(model.layers):
            if physical_intermediate != intermediate_size:
                for name, shape in (("gate_proj", (physical_intermediate, hidden_size)),
                                    ("up_proj", (physical_intermediate, hidden_size)),
                                    ("down_proj", (hidden_size, physical_intermediate))):
                    projection = getattr(layer.mlp, name)
                    if ((projection.out_features, projection.in_features) != shape
                            or tuple(projection.weight.shape) != shape
                            or projection.weight.dtype != torch.float16):
                        raise ValueError(f"decoder layer {index} {name} physical MLP shape differs from cold padding")
            for container, names, cores in (
                (layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj"), topology.attn_tp),
                (layer.mlp, ("gate_proj", "up_proj", "down_proj"), topology.mlp_tp),
            ):
                for name in names:
                    projection = getattr(container, name)
                    partition = 0 if name in {"o_proj", "down_proj"} else 1
                    if (getattr(projection, "_rpu_linear_num_cores", None) != cores or
                            getattr(projection, "_rpu_linear_partition", None) != partition):
                        raise ValueError(f"decoder layer {index} {name} weight layout differs from the cold topology")

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
            # Newer HF Qwen3 rotary modules expose max_seq_len_cached/config,
            # not max_position_embeddings directly. Falling back to 4096 here
            # built a truncated table for a 40960-position model: P4096 prefill
            # was correct, then decode position 4096 read past the table and
            # produced a confidently wrong token. Read every supported spelling
            # and keep the largest authoritative capacity.
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

    # R-Phase 1 (Qwen3-VL): pass mrope_section as the trailing append-only
    # kwarg. Empty list for non-mrope archs preserves legacy behavior.
    mrope_section_arg = list(mrope_section) if mrope_section else []
    # R-Phase 2 (Qwen3-VL): same append-only contract for
    # deepstack_lang_layers. Empty default = no DeepStack injection
    # (transparent for Qwen3 / Llama and for Qwen3-VL text-only smoke tests).
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

    # Own a GraphCache for the patched Qwen3 / Llama / Qwen3-VL text-only
    # forward. Capture groups the decoder dispatches into a reusable batch
    # instead of submitting each kernel through the immediate path.
    #
    # The Qwen3-VL e2e flow (adapters/qwen3_vl/__init__.py) bypasses this
    # patched forward and uses its own `_rpu_text_graph_cache` external
    # wrap; the cache stashed here stays dormant for that path (small
    # one-off memory cost — no second BUILD because forward never enters
    # this closure).
    from contextlib import nullcontext as _nullcontext
    import rpu_backend as _rb
    decoder_graph_cache = _new_decoder_graph_cache(
        topology, max_entries=_graph_cache_max_entries)
    decoder_batch_decode_enabled = (
        arch == "qwen3" and (topology is None or topology.num_cores == 8)
        and not any(weight.dtype == torch.uint8 for weight in q_w_list)
        and not (num_layers == 64 and hidden_size == 5120))
    decoder_num_layers = int(num_layers)
    decoder_hidden_size = int(hidden_size)
    # FNV1a-style hash tiebreaker so plain qwen3 (empty list → 0) and
    # qwen3-vl text-only smoke (non-empty layer list) cannot alias on a
    # shared model instance. Mirrors adapters/qwen3_vl/text.py:194-198.
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
        decode_plan = None
        batched_prefill = _is_batched_prefill(input_ids, inputs_embeds)
        if isinstance(past_key_values, _RPUCache_for_sig):
            validate_decoder_cache_topology(topology, past_key_values)
            if inputs_embeds is not None:
                _batch, _seq_len = (int(inputs_embeds.shape[0]),
                                    int(inputs_embeds.shape[1]))
            elif input_ids is not None:
                _batch, _seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
            else:
                _batch, _seq_len = 1, 0  # runner will raise on XOR violation
            validate_decoder_cache_topology(topology, past_key_values, batch_size=_batch)
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
                _execution_len, _planned_chunk_size = prefill_plan[:2]
            elif _seq_len == 1:
                decode_plan = _text_decode_execution_plan(self, self._rpu_decoder_handle)
            # P3 — position dropped from sig so successive decode steps hit
            # the same cached entry (first step BUILD, rest REPLAY via P2's
            # sync-only fast path). Safe because every position-derived
            # kernel arg flows through set_regs + add_kernel_mutable and is
            # refreshed by sync_mutable_params on each REPLAY.
            # Batch, logical length and physical execution plan all key the
            # graph: each changes either dispatch count or baked SPM offsets.
            def _capture(batch, seq_len, plan=None):
                execution_len, planned_chunk_size = (
                    (int(plan[0]), int(plan[1]))
                    if plan is not None else (int(seq_len), 0)
                )
                plan_key = (
                    plan[2].graph_key_words()
                    if plan is not None
                    else (decode_plan.graph_key_words() if int(seq_len) == 1 else ())
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
                        *(topology.identity() if topology is not None else ()),
                        int(getattr(self, "_rpu_execution_generation", 0)),
                        chunk_policy_key(self._rpu_decoder_handle),
                        planned_chunk_size,
                        prefill_position_key(
                            execution_len, past_key_values.position
                        ),
                        *plan_key,
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

    from rpu_backend.api._execution import execution_serialized
    rpu_decoder_model_forward = execution_serialized(
        rpu_decoder_model_forward
    )

    # 2. Configure and tentatively publish the replacement while retaining a
    # complete snapshot of the old Python state. The old native handle remains
    # live until every Python assignment succeeds. Publication or old-handle
    # retirement failure restores the snapshot. An unsafe retirement retains
    # the pending handle too: no more native work is allowed in that process.
    handle = torch.ops.rpu.causal_decoder_create(
        rpu_env_bool("RPU_QWEN3_SPM_KV_BY_MHA", default=True),
        rpu_env_bool("RPU_ADARMS_FUSED_BCAST"),
        **({"num_cores": topology.num_cores, "vocab_size": model.config.vocab_size}
           if topology is not None else {}),
    )
    resource = _InstalledNativeResource(
        model, handle, torch.ops.rpu.causal_decoder_destroy,
        graphs=(getattr(model, "_rpu_text_graph_cache", None), decoder_graph_cache),
        keepalive=set_weights_args, label="decoder",
        handle_name="_rpu_decoder_handle")
    handle_finalizer = resource.finalizer
    try:
        set_weights_op(handle, *set_weights_args)
        if topology is not None:
            native_topology = tuple(torch.ops.rpu.causal_decoder_topology(handle))
            if native_topology != topology.identity()[1:]:
                raise RuntimeError(
                    "native decoder topology differs from the bound weight/cache layout: "
                    f"actual={native_topology}, expected={topology.identity()[1:]}")
        # MR-A: declare the certified chunk envelope BEFORE the first forward.
        # Deny-by-default — the C++ planner refuses to prefill an undeclared
        # handle, because the auto search can otherwise pick a chunk above the
        # model's true SPM ceiling and WEDGE the board.
        #
        # The TABLE is not here: DAG-03 forbids arch tokens in runtime/, and it is
        # real arch coupling (scripts/check_import_graph_v5.py:61 rejected an
        # earlier draft that put it in runtime/chunk_envelope.py). The caller hands
        # us the lookup; we own only the geometry, which we already derived above.
        _envelope = chunk_envelope_for(
            arch, decoder_num_layers, decoder_hidden_size)
        torch.ops.rpu.causal_decoder_set_chunk_envelope(
            handle, int(_envelope[0]), int(_envelope[1]))

        _prefill_cfg = execution_config.get("prefill", {})
        if arch == "qwen3":
            torch.ops.rpu.causal_decoder_set_linear_acc32(
                handle, bool(_prefill_cfg.get("linear_acc32", False)))
        _requested_chunk = _prefill_cfg.get("chunk_size", "auto")
        torch.ops.rpu.causal_decoder_set_chunk_size_override(
            handle,
            0 if _requested_chunk == "auto" else int(_requested_chunk),
        )
        decode_stage_descriptor = tuple(
            torch.ops.rpu.causal_decoder_resolve_decode_stage_descriptor(
                handle
            )
        )
        if not decode_stage_descriptor:
            raise RuntimeError(
                "causal decoder decode planner returned an empty descriptor"
            )

        model._rpu_deepstack_lang_layers = deepstack_lang_layers_arg
        model._rpu_batch_decode_enabled = decoder_batch_decode_enabled
        model._rpu_decoder_graph_cache = decoder_graph_cache
        model._rpu_decoder_num_layers = decoder_num_layers
        model._rpu_decoder_hidden_size = decoder_hidden_size
        model._rpu_decoder_deepstack_hash = _ds_hash
        if topology is not None and not hasattr(model, "_rpu_decoder_topology"):
            model._rpu_decoder_topology = topology
        model._rpu_decode_stage_descriptor = decode_stage_descriptor
        model._rpu_decoder_handle = handle
        model._rpu_decoder_retirement_state = resource
        model._rpu_decoder_handle_finalizer = handle_finalizer
        model._rpu_prefill_execution_alignment = (
            16 if arch == "qwen3_vl_text" else 1
        )
        model._rpu_execution = execution_config

        import types
        model.forward = types.MethodType(rpu_decoder_model_forward, model)

        if old_resource is not None:
            old_resource.retire()
            if old_resource.parent is not None:
                resource.take_ownership(old_resource.parent())
        elif old_handle is not _missing and old_handle is not None:
            raise RuntimeError("decoder replacement requires its actual retirement state")
    except BaseException as error:
        resource.cleanup_failure(error, model, old_install_state, old_instance_forward)
        # Rollback must not re-enter the attribute hook that just failed.
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
        raise

    # The old callback must not outlive its retired handle. The current native
    # registry uses monotonic IDs, but detaching still preserves one callback
    # per handle and avoids a redundant not-found destroy at model GC.
    if old_finalizer is not None and getattr(old_finalizer, "alive", False):
        old_finalizer.detach()

    _LOG.info("Patched %s (instance) with all-layers-once fused forward, "
              "handle=%d, num_layers=%d",
              type(model).__name__, handle, num_layers)

    return handle
