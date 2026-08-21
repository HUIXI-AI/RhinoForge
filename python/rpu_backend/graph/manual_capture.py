"""Manual capture frontend.

This module provides a thin Python wrapper around
`torch.rpu.GraphCache.capture(sig)` that hides
SignaturePlanner construction, exposes a CUDA-Graph-style
`g.replay(*args)` entry, and refuses to silently recapture on
signature drift.

Boundary against the Dynamo frontend (`dynamo_backend.py`):

  - No `RpuDynamoBackend` / `_RpuCompiledModule` dependency.
  - No `EntryRegistry` / `GraphPlan` dependency (they exist for the
    Dynamo path's frontend LRU routing; the manual path uses
    one cache entry per `GraphCapture` instance).
  - No `torch.compile` / `torch.fx` involvement.
  - Reuses `SignaturePlanner` for sig construction so the two
    frontends share signature semantics.

Fast replay via `dynamic_inputs` is not implemented because it requires
backend RECORDING-time patch identification. Non-empty `dynamic_inputs`
raises `NotImplementedError`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .admission import SignaturePlanner


class GraphCapture:
    """Manual frontend for one `(callable, example_inputs)` pair.

    Eager-RECORDING semantics: `__init__` opens a cache scope, runs
    `callable(*example_args, **example_kwargs)` once to capture the
    graph, and stores the captured output. After construction, the
    backend is in BUILT (or PASSTHROUGH if `non_replayable_reason`
    fired during `end()`).

    Replay semantics: `g.replay(*args, **kwargs)` rebuilds the
    signature from the new arguments and **refuses to proceed** if
    the signature differs from capture-time. Manual frontend does
    not auto-recapture; the user must call `invalidate()` and
    construct a new `GraphCapture` if shapes change. This is the
    deliberate boundary against the Dynamo frontend, which absorbs
    shape drift via guard-driven retrace.

    The cache entry's lifetime is tied to this `GraphCapture`'s
    Python reference. `del g` (or eviction via `g.invalidate()`)
    triggers backend graph destruction and SPM release.
    """

    def __init__(
        self,
        callable_: Callable[..., Any],
        example_args: Sequence[Any] = (),
        example_kwargs: Optional[Dict[str, Any]] = None,
        *,
        dynamic_inputs: Optional[List[str]] = None,
        cache: Optional["torch.rpu.GraphCache"] = None,
        planner: Optional[SignaturePlanner] = None,
        op_id: Optional[Any] = None,
    ):
        if dynamic_inputs:
            raise NotImplementedError(
                "GraphCapture: dynamic_inputs requires fast replay. "
                "It requires backend RECORDING-time patch identification "
                "and is not implemented."
            )
        if example_kwargs is None:
            example_kwargs = {}
        if not isinstance(example_args, (tuple, list)):
            example_args = (example_args,)
        if not callable(callable_):
            raise TypeError(
                f"GraphCapture: callable_ must be callable, got "
                f"{type(callable_).__name__}"
            )

        self._callable = callable_
        self._planner = planner or SignaturePlanner()
        self._cache = cache if cache is not None else torch.rpu.GraphCache()
        self._example_args = tuple(example_args)
        self._example_kwargs = dict(example_kwargs)

        # op_id derivation: defaults to id(callable_) so two GraphCapture
        # instances over the same module share a cache entry when shapes
        # also match (resource sharing). User can override with a hashable
        # marker to force separation.
        if op_id is None:
            op_id_int = id(callable_)
        elif isinstance(op_id, int):
            op_id_int = int(op_id)
        else:
            op_id_int = hash(op_id)

        # Sig construction: positional + kwarg values flattened in the
        # same order `SignaturePlanner.build_signature` walks them.
        all_args = self._example_args + tuple(self._example_kwargs.values())
        self._sig = self._planner.build_signature(op_id_int, all_args)
        self._sig_key = SignaturePlanner.signature_key(self._sig)
        self._kwarg_keys = tuple(self._example_kwargs.keys())

        # Eager RECORDING. cache.capture(sig) handles begin/end/touch_entry.
        with self._cache.capture(self._sig):
            self._cached_output = self._callable(
                *self._example_args, **self._example_kwargs
            )
        # state -> BUILT (or PASSTHROUGH if non_replayable_reason fired).
        self._captured = True

    # ----- replay -----

    def replay(self, *args: Any, **kwargs: Any) -> Any:
        """Re-run the captured graph with new inputs.

        Raises RuntimeError on signature drift. The diagnostic names
        which signature layer first diverged (GmId / TensorStruct /
        ShapeBucket / DtypeLayout / ScalarFlags / BranchKey /
        SegmentKey) so the caller knows which input changed.
        """
        if not self._captured:
            raise RuntimeError(
                "GraphCapture.replay called before capture completed; "
                "construction must have raised."
            )

        # Kwarg names must match capture-time names; values are canonicalized
        # to capture-time order before rebuilding the signature so callers may
        # pass kwargs in any order without causing false drift.
        if set(kwargs.keys()) != set(self._kwarg_keys):
            raise RuntimeError(
                f"GraphCapture.replay: kwarg keys differ. Capture-time "
                f"{self._kwarg_keys!r}, replay {tuple(kwargs.keys())!r}."
            )

        # Rebuild sig from current args; refuse on drift (no auto-recapture).
        ordered_kwargs = tuple(kwargs[k] for k in self._kwarg_keys)
        all_current = tuple(args) + ordered_kwargs
        current_sig = self._planner.build_signature(
            self._sig._impl.op_id, all_current
        )
        if SignaturePlanner.signature_key(current_sig) != self._sig_key:
            miss_layer = SignaturePlanner.attribute_miss(
                current_sig, self._sig
            ) or "<unknown>"
            raise RuntimeError(
                f"GraphCapture.replay: signature mismatch (first diverge at "
                f"layer {miss_layer!r}). Manual frontend does not "
                f"auto-recapture. Call .invalidate() and construct a new "
                f"GraphCapture. Automatic shape-driven retrace is unsupported."
            )

        with self._cache.capture(self._sig):
            out = self._callable(*args, **kwargs)
        return out

    __call__ = replay  # ergonomic alias: g(*args) <-> g.replay(*args)

    # ----- lifecycle -----

    def invalidate(self) -> None:
        """Evict the backend cache entry; SPM and prepared queues are
        released. The `GraphCapture` becomes unusable; further `replay`
        calls will fail at backend admission. Construct a new
        `GraphCapture` if you want to capture again."""
        self._cache.evict(self._sig)
        self._captured = False

    # ----- diagnostics -----

    @property
    def signature(self) -> "torch.rpu.GraphSignature":
        return self._sig

    @property
    def cache(self) -> "torch.rpu.GraphCache":
        return self._cache

    @property
    def cached_output(self) -> Any:
        """Output produced by the capture-time call. The first replay
        does NOT reuse this — backend re-allocates output each call
        unless fast replay is wired."""
        return self._cached_output

    def replayable(self) -> bool:
        """True iff the backend graph is in BUILT state without a
        non_replayable_reason. False for PASSTHROUGH-degraded entries
        (e.g., RPU strided copy in the captured op stream)."""
        entry = self._snapshot_entry()
        if entry is None:
            return False
        return not entry.non_replayable_reason

    def non_replayable_reason(self) -> str:
        """Empty string if replayable; otherwise the backend's
        actionable diagnostic (e.g. `rpu_copy: RPU strided copy in
        graph uses eager sync_point; replay unsupported`)."""
        entry = self._snapshot_entry()
        return entry.non_replayable_reason if entry is not None else ""

    def stats(self) -> Dict[str, Any]:
        """Return per-entry counters for this capture. Useful in
        decode loops to confirm `replay_count` advances and
        `recapture_count` stays at 0."""
        entry = self._snapshot_entry()
        if entry is None:
            return {}
        return {
            "kernel_count": entry.kernel_count,
            "segment_count": entry.segment_count,
            "data_node_count": entry.data_node_count,
            "boundary_flush_count": entry.boundary_flush_count,
            "replay_count": entry.replay_count,
            "recapture_count": entry.recapture_count,
            "host_callback_tier1_count": entry.host_callback_tier1_count,
            "host_callback_tier2_count": entry.host_callback_tier2_count,
            "host_callback_tier3_count": entry.host_callback_tier3_count,
            "tier3_oneshot_count": entry.tier3_oneshot_count,
            "register_census_present": entry.register_census_present,
            "register_census_build_committed":
                entry.register_census_build_committed,
            "register_census_policy_fingerprint":
                entry.register_census_policy_fingerprint,
            "register_census_policy_digest":
                entry.register_census_policy_digest,
            "register_census_plan_hash": entry.register_census_plan_hash,
            "register_census_live_epoch": entry.register_census_live_epoch,
            "register_census_node_count": entry.register_census_node_count,
            "register_census_dma_count": entry.register_census_dma_count,
            "register_census_barrier_count":
                entry.register_census_barrier_count,
            "register_census_kernel_count": entry.register_census_kernel_count,
            "register_census_typed_kernel_count":
                entry.register_census_typed_kernel_count,
            "register_census_no_ddr_kernel_count":
                entry.register_census_no_ddr_kernel_count,
            "register_census_operand_count":
                entry.register_census_operand_count,
            "register_census_weight_operand_count":
                entry.register_census_weight_operand_count,
            "register_census_key_cache_operand_count":
                entry.register_census_key_cache_operand_count,
            "register_census_value_cache_operand_count":
                entry.register_census_value_cache_operand_count,
            "register_census_semantic_input_operand_count":
                entry.register_census_semantic_input_operand_count,
            "register_census_ordered_node_kind_hash":
                entry.register_census_ordered_node_kind_hash,
            "register_census_ordered_kernel_hash":
                entry.register_census_ordered_kernel_hash,
            "non_replayable_reason": entry.non_replayable_reason,
        }

    def _snapshot_entry(self):
        """Locate this capture's entry in the cache snapshot list.
        Returns None when the entry has been evicted."""
        try:
            snap = self._cache.snapshot()
        except Exception:
            return None
        target = self._sig._impl
        for entry in snap:
            if entry.signature == target:
                return entry
        return None

    def __repr__(self) -> str:
        state = "captured" if self._captured else "invalidated"
        try:
            n_kernels = self.stats().get("kernel_count", 0)
        except Exception:
            n_kernels = 0
        return (
            f"GraphCapture({state}, sig_key={self._sig_key[0]:x}..., "
            f"kernels={n_kernels})"
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def capture(
    callable_: Callable[..., Any],
    example_inputs: Any = None,
    *,
    args: Optional[Sequence[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_inputs: Optional[List[str]] = None,
    cache: Optional["torch.rpu.GraphCache"] = None,
    planner: Optional[SignaturePlanner] = None,
    op_id: Optional[Any] = None,
) -> GraphCapture:
    """Build and eagerly RECORDING-capture a graph for `callable_`.

    Two ways to specify capture-time inputs:

      Positional convenience::

          rpu.capture(model, input_ids)               # single tensor
          rpu.capture(model, (input_ids, mask))       # tuple of args

      Detailed (matches `torch.export.export(model, args=, kwargs=)`)::

          rpu.capture(model, args=(input_ids,),
                      kwargs={"past_key_values": cache})

    `dynamic_inputs` is reserved for fast replay and raises
    `NotImplementedError` if non-empty.

    `cache` defaults to a fresh `torch.rpu.GraphCache()` so two
    `GraphCapture` instances are independent unless the caller
    passes a shared one explicitly. Mixing manual and Dynamo
    frontends on the same cache is supported but not recommended because
    they use distinct admission and lifetime policies.
    """
    if args is not None or kwargs is not None:
        if example_inputs is not None:
            raise ValueError(
                "rpu.capture: pass either positional `example_inputs` or "
                "keyword `args=/kwargs=`, not both."
            )
        eff_args: Tuple[Any, ...] = tuple(args) if args is not None else ()
        eff_kwargs: Dict[str, Any] = dict(kwargs) if kwargs is not None else {}
    else:
        if example_inputs is None:
            eff_args = ()
        elif isinstance(example_inputs, (tuple, list)):
            eff_args = tuple(example_inputs)
        else:
            eff_args = (example_inputs,)
        eff_kwargs = {}

    return GraphCapture(
        callable_,
        eff_args,
        eff_kwargs,
        dynamic_inputs=dynamic_inputs,
        cache=cache,
        planner=planner,
        op_id=op_id,
    )
