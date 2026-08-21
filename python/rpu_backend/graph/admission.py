"""Graph admission frontend: plans, signatures, miss attribution, and LRU.

This module owns the Python-side graph admission pipeline consumed by
the Dynamo and manual graph frontends:

  - GraphPlan descriptors for one graph call
  - tensor structure encoding
  - dtype / layout encoding
  - literal shape encoding
  - scalar flags hashing
  - branch_key derivation from the user-provided callable
  - segment_key derivation
  - workload_key inference (decode / prefill / generic)
  - cache miss reason attribution: which signature layer first diverged
  - EntryRegistry's frontend LRU in front of backend GraphCache

Backend remains the final admission authority. Miss attribution wraps
the existing C++ `LayeredSignature.first_diverge` API
(`src/graph/graph_infra.h`).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# GraphPlan vocabulary.
# ---------------------------------------------------------------------------

WORKLOAD_PREFILL = "prefill"
WORKLOAD_DECODE = "decode"
WORKLOAD_GENERIC = "generic"
WORKLOAD_KINDS = (WORKLOAD_PREFILL, WORKLOAD_DECODE, WORKLOAD_GENERIC)

ADMISSION_RECORD = "record"
ADMISSION_REPLAY = "replay"
ADMISSION_RECAPTURE = "recapture"
ADMISSION_KINDS = (ADMISSION_RECORD, ADMISSION_REPLAY, ADMISSION_RECAPTURE)

LIFECYCLE_RETAIN = "retain"
LIFECYCLE_EVICT = "evict"
LIFECYCLE_RECAPTURE = "recapture"
LIFECYCLE_KINDS = (LIFECYCLE_RETAIN, LIFECYCLE_EVICT, LIFECYCLE_RECAPTURE)

MISS_NO_ENTRY = "no_entry"
MISS_GM_ID = "gm_id"
MISS_TENSOR_STRUCT = "tensor_struct"
MISS_SHAPE_BUCKET = "shape_bucket"
MISS_DTYPE_LAYOUT = "dtype_layout"
MISS_SCALAR_FLAGS = "scalar_flags"
MISS_BRANCH_KEY = "branch_key"
MISS_SEGMENT_KEY = "segment_key"
MISS_REASONS = (
    MISS_NO_ENTRY,
    MISS_GM_ID,
    MISS_TENSOR_STRUCT,
    MISS_SHAPE_BUCKET,
    MISS_DTYPE_LAYOUT,
    MISS_SCALAR_FLAGS,
    MISS_BRANCH_KEY,
    MISS_SEGMENT_KEY,
)


@dataclass
class GraphPlan:
    """Frontend strategy descriptor for one traced graph call."""

    signature: Any
    guard_key: int
    entry_key: int
    workload_kind: str = WORKLOAD_GENERIC
    expected_admission: str = ADMISSION_RECORD
    lifecycle_intent: str = LIFECYCLE_RETAIN
    branch_key: int = 0
    segment_key: int = 0

    def __post_init__(self) -> None:
        if self.workload_kind not in WORKLOAD_KINDS:
            raise ValueError(
                f"GraphPlan.workload_kind must be one of {WORKLOAD_KINDS}, "
                f"got {self.workload_kind!r}")
        if self.expected_admission not in ADMISSION_KINDS:
            raise ValueError(
                f"GraphPlan.expected_admission must be one of "
                f"{ADMISSION_KINDS}, got {self.expected_admission!r}")
        if self.lifecycle_intent not in LIFECYCLE_KINDS:
            raise ValueError(
                f"GraphPlan.lifecycle_intent must be one of "
                f"{LIFECYCLE_KINDS}, got {self.lifecycle_intent!r}")

    def to_log_str(self) -> str:
        """One-line summary for frontend cache diagnostic logs."""
        return (
            f"GraphPlan("
            f"workload={self.workload_kind}, "
            f"admission={self.expected_admission}, "
            f"lifecycle={self.lifecycle_intent}, "
            f"guard={self.guard_key & 0xFFFFFFFF:08x}, "
            f"entry={self.entry_key & 0xFFFFFFFF:08x}, "
            f"branch={self.branch_key & 0xFFFFFFFF:08x}, "
            f"segment={self.segment_key & 0xFFFFFFFF:08x})"
        )


def build_plan_for_call(
    signature: Any,
    *,
    workload_kind: str = WORKLOAD_GENERIC,
    expected_admission: str = ADMISSION_RECORD,
    lifecycle_intent: str = LIFECYCLE_RETAIN,
    branch_key: int = 0,
    segment_key: int = 0,
) -> GraphPlan:
    """Build a GraphPlan and derive guard/entry keys from `signature`."""
    impl = getattr(signature, "_impl", signature)
    op_id = getattr(impl, "op_id", 0)
    shapes = tuple(getattr(impl, "shapes", ()))
    dtypes = tuple(getattr(impl, "dtypes", ()))
    flags = getattr(impl, "flags", 0)
    sig_branch_key = getattr(impl, "branch_key", branch_key)
    sig_segment_key = getattr(impl, "segment_key", segment_key)

    key = hash((op_id, shapes, dtypes, flags,
                sig_branch_key, sig_segment_key)) & 0x7FFFFFFFFFFFFFFF

    return GraphPlan(
        signature=signature,
        guard_key=key,
        entry_key=key,
        workload_kind=workload_kind,
        expected_admission=expected_admission,
        lifecycle_intent=lifecycle_intent,
        branch_key=sig_branch_key,
        segment_key=sig_segment_key,
    )


# ---------------------------------------------------------------------------
# Constants used by build_signature().
# ---------------------------------------------------------------------------

_DTYPE_CODE = {
    torch.float16: 5, torch.bfloat16: 15, torch.float32: 6,
    torch.int32: 3, torch.int64: 4, torch.bool: 11,
}

_TAG_TENSOR = -1
_TAG_LIST = -2
_TAG_TUPLE = -3
_TAG_DICT = -4
_TAG_END = -5

_HASH_MASK = 0x7FFFFFFFFFFFFFFF


def _fnv1a_64_u(s: str) -> int:
    h = 0xcbf29ce484222325
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h & _HASH_MASK


def _branch_key_token(value: Any) -> str:
    if value is None:
        return "n:"
    if isinstance(value, bool):
        return f"b:{int(value)}"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float):
        return f"f:{value.hex()}"
    if isinstance(value, str):
        return f"s:{len(value)}:{value}"
    if isinstance(value, tuple):
        return "t:[" + ",".join(_branch_key_token(v) for v in value) + "]"
    if isinstance(value, list):
        return "l:[" + ",".join(_branch_key_token(v) for v in value) + "]"
    raise TypeError(
        "branch_key_fn must return None, bool, int, float, str, tuple, or "
        f"list; got {type(value).__name__}"
    )


def _stable_branch_key(value: Any) -> int:
    if value is None:
        return 0
    return _fnv1a_64_u(_branch_key_token(value))


def _visit(shapes: List[int], dtypes: List[int], flags_box: List[int],
           x: Any) -> None:
    if isinstance(x, torch.Tensor):
        shapes.append(_TAG_TENSOR)
        shapes.append(x.dim())
        shapes.extend(int(s) for s in x.shape)
        dtypes.append(_DTYPE_CODE.get(x.dtype, 0))
    elif isinstance(x, list):
        shapes.append(_TAG_LIST)
        shapes.append(len(x))
        for v in x:
            _visit(shapes, dtypes, flags_box, v)
        shapes.append(_TAG_END)
    elif isinstance(x, tuple):
        shapes.append(_TAG_TUPLE)
        shapes.append(len(x))
        for v in x:
            _visit(shapes, dtypes, flags_box, v)
        shapes.append(_TAG_END)
    elif isinstance(x, dict):
        shapes.append(_TAG_DICT)
        shapes.append(len(x))
        for k in sorted(x.keys(), key=lambda y: repr(y)):
            flags_box[0] ^= hash(("dk", repr(k))) & _HASH_MASK
            _visit(shapes, dtypes, flags_box, x[k])
        shapes.append(_TAG_END)
    elif isinstance(x, bool):  # bool first; bool is a subclass of int
        flags_box[0] ^= hash(("b",)) & _HASH_MASK
    elif isinstance(x, int):
        flags_box[0] ^= hash(("i",)) & _HASH_MASK
    elif isinstance(x, float):
        flags_box[0] ^= hash(("f",)) & _HASH_MASK
    elif isinstance(x, str):
        flags_box[0] ^= hash(("s",)) & _HASH_MASK
    elif x is None:
        flags_box[0] ^= hash(("N",)) & _HASH_MASK
    elif isinstance(x, torch.dtype):
        flags_box[0] ^= hash(("dt", _DTYPE_CODE.get(x, repr(x)))) & _HASH_MASK
    elif isinstance(x, torch.device):
        flags_box[0] ^= hash(("dv", repr(x))) & _HASH_MASK
    else:
        flags_box[0] ^= hash((type(x).__name__, id(x))) & _HASH_MASK


# ---------------------------------------------------------------------------
# SignaturePlanner
# ---------------------------------------------------------------------------

class SignaturePlanner:
    """Centralised signature construction + miss attribution.

    A single instance is stateless and may be shared across backends;
    no per-instance configuration is required.
    """

    def __init__(self, *, debug: bool = False):
        self._debug = debug

    # ----- guard / signature construction -----

    def build_signature(
        self,
        gm_id: int,
        args: Sequence[Any],
        *,
        branch_key: int = 0,
        segment_key: int = 0,
    ) -> "torch.rpu.GraphSignature":
        """Encode (gm_id, args) into a GraphSignature.

        Exact algorithm:

          op_id  = gm_id & 0x7FFFFFFFFFFFFFFF
          shapes = recursive walk of args, emitting tag/dim/value tokens
          dtypes = per-tensor dtype code (in walk order)
          flags  = XOR of hashed scalar / dict-key / opaque-instance hits
          branch_key / segment_key = caller-supplied (defaults 0).
        """
        sig = torch.rpu.GraphSignature()
        sig._impl.op_id = gm_id & _HASH_MASK
        shapes: List[int] = []
        dtypes: List[int] = []
        flags_box = [0]

        for a in args:
            _visit(shapes, dtypes, flags_box, a)

        sig._impl.shapes = shapes
        sig._impl.dtypes = dtypes
        sig._impl.flags = int(flags_box[0] & _HASH_MASK)
        sig._impl.branch_key = int(branch_key) & _HASH_MASK
        sig._impl.segment_key = int(segment_key) & _HASH_MASK
        return sig

    @staticmethod
    def stable_branch_key(value: Any) -> int:
        """Deterministically hash a branch_key_fn return value into uint63.

        None maps to 0. Tagged hashing keeps `False / 0 / "0"` distinct,
        and lists are tagged differently from tuples so structurally
        different inputs cannot collide.
        """
        return _stable_branch_key(value)

    @staticmethod
    def signature_key(sig: "torch.rpu.GraphSignature") -> Tuple:
        """Return a hashable Python tuple for LRU / dict keys.

        Avoids relying on pybind11's `__hash__`, which is not stable
        across rebuilds because the C++ impl pointer participates.
        """
        impl = sig._impl
        return (impl.op_id, tuple(impl.shapes), tuple(impl.dtypes),
                impl.flags, impl.branch_key, impl.segment_key)

    # ----- workload kind inference -----

    @staticmethod
    def infer_workload_kind(
        args: Sequence[Any],
        *,
        decode_seq_len: int = 1,
    ) -> str:
        """Heuristic decoder: examine the first tensor arg's shape.

        Returns:
          - WORKLOAD_DECODE  if the first tensor has shape `[B, decode_seq_len]`
            (typically `[1, 1]` for single-token decode).
          - WORKLOAD_PREFILL if the first tensor has shape `[B, S]` with
            `S > decode_seq_len`.
          - WORKLOAD_GENERIC if no tensor arg is found.

        Backed by the same decode/prefill shape convention used by the
        Dynamo backend.
        """
        for a in args:
            if isinstance(a, torch.Tensor):
                if a.dim() >= 2:
                    seq_len = int(a.shape[1])
                    if seq_len == decode_seq_len:
                        return WORKLOAD_DECODE
                    if seq_len > decode_seq_len:
                        return WORKLOAD_PREFILL
                return WORKLOAD_GENERIC
        return WORKLOAD_GENERIC

    # ----- plan construction -----

    def make_plan(
        self,
        gm_id: int,
        args: Sequence[Any],
        *,
        branch_key: int = 0,
        segment_key: int = 0,
        expected_admission: str = ADMISSION_RECORD,
        lifecycle_intent: str = LIFECYCLE_RETAIN,
        workload_kind: Optional[str] = None,
    ) -> GraphPlan:
        """One-shot helper: build signature + GraphPlan in one call.

        `workload_kind` defaults to the planner's inferred kind.
        """
        sig = self.build_signature(
            gm_id, args, branch_key=branch_key, segment_key=segment_key)
        if workload_kind is None:
            workload_kind = self.infer_workload_kind(args)
        plan = build_plan_for_call(
            sig,
            workload_kind=workload_kind,
            expected_admission=expected_admission,
            lifecycle_intent=lifecycle_intent,
            branch_key=branch_key,
            segment_key=segment_key,
        )
        return plan

    # ----- miss reason attribution -----

    @staticmethod
    def attribute_miss(
        sig_a: "torch.rpu.GraphSignature",
        sig_b: "torch.rpu.GraphSignature",
    ) -> Optional[str]:
        """Return the name of the first SignatureLayer that diverges.

        Wraps the C++ `LayeredSignature::first_diverge` API. Layers in
        order: GmId -> TensorStruct -> ShapeBucket -> DtypeLayout ->
        ScalarFlags -> BranchKey -> SegmentKey. Returns None when the
        two signatures match in all layers (no miss to attribute).
        """
        layered_a = torch.rpu.layered_from(sig_a._impl)
        layered_b = torch.rpu.layered_from(sig_b._impl)
        layer = layered_a.first_diverge(layered_b)
        if int(layer) >= int(torch.rpu.SignatureLayer.kCount):
            return None
        return torch.rpu.signature_layer_name(layer)

    @staticmethod
    def attribute_miss_reason(
        sig_a: "torch.rpu.GraphSignature",
        sig_b: "torch.rpu.GraphSignature",
    ) -> Optional[str]:
        """Typed variant of `attribute_miss`.

        Returns one of this module's `MISS_*` constants identifying
        which `LayeredSignature` layer first diverges, mapped to the
        retained miss-reason vocabulary. Returns `None` when both
        signatures match in all layers (no miss to attribute).

        Use this from `EntryRegistry` and benchmark-side diagnostic
        code where typed comparison is required. Use `attribute_miss`
        when the human-readable layer name (e.g. `"ShapeBucket"`)
        is wanted instead.
        """
        layered_a = torch.rpu.layered_from(sig_a._impl)
        layered_b = torch.rpu.layered_from(sig_b._impl)
        layer = layered_a.first_diverge(layered_b)
        if int(layer) >= int(torch.rpu.SignatureLayer.kCount):
            return None
        return _LAYER_TO_MISS_REASON.get(int(layer))


# Layer-int -> MISS_* mapping. Built from raw int values matching the C++
# `SignatureLayer` enum order in `src/graph/graph_infra.h`:
#   GmId=0, TensorStruct=1, ShapeBucket=2, DtypeLayout=3, ScalarFlags=4,
#   BranchKey=5, SegmentKey=6.
# Built at import time as a plain int -> str dict; this avoids touching
# `torch.rpu.SignatureLayer` here because rpu_backend.__init__ has not finished
# wiring up `torch.rpu` when admission.py is imported.
_LAYER_TO_MISS_REASON = {
    0: MISS_GM_ID,
    1: MISS_TENSOR_STRUCT,
    2: MISS_SHAPE_BUCKET,
    3: MISS_DTYPE_LAYOUT,
    4: MISS_SCALAR_FLAGS,
    5: MISS_BRANCH_KEY,
    6: MISS_SEGMENT_KEY,
}


# ---------------------------------------------------------------------------
# EntryRegistry: frontend LRU policy in front of backend GraphCache.
# ---------------------------------------------------------------------------

STATE_PASSTHROUGH = 0
STATE_RECORDING = 1
STATE_BUILT = 2
STATE_REPLAYING = 3


def _sig_impl(sig: Any) -> Any:
    """Unwrap a Python GraphSignature to the C++ pybind impl if needed."""
    return getattr(sig, "_impl", sig)


class _PoolEntry:
    __slots__ = ("signature", "plan", "last_state", "replay_count",
                 "recapture_count")

    def __init__(self, signature: Any, plan: GraphPlan):
        self.signature = signature
        self.plan = plan
        self.last_state: int = STATE_PASSTHROUGH
        self.replay_count: int = 0
        self.recapture_count: int = 0


class EntryRegistry:
    """Single-LRU frontend cache routing in front of backend GraphCache.

    The registry owns no RPU memory. It chooses which backend GraphCache entry
    should receive a call, enforces a soft LRU budget, and records frontend
    replay/recapture/evict diagnostics.
    """

    def __init__(
        self,
        cache: "torch.rpu.GraphCache",
        *,
        max_entries: Optional[int] = None,
        prefill_max_entries: Optional[int] = None,
        decode_max_entries: Optional[int] = None,
        generic_max_entries: Optional[int] = None,
        planner: Optional[SignaturePlanner] = None,
    ):
        self._cache = cache
        self._planner = planner or SignaturePlanner()
        if max_entries is None:
            max_entries = (
                decode_max_entries
                if decode_max_entries is not None
                else generic_max_entries
                if generic_max_entries is not None
                else prefill_max_entries
                if prefill_max_entries is not None
                else 4
            )
        self._max_entries = int(max_entries)
        if self._max_entries < 1:
            raise ValueError(
                f"EntryRegistry.max_entries must be >= 1, got "
                f"{self._max_entries}"
            )
        self._pool: "OrderedDict[Tuple, _PoolEntry]" = OrderedDict()
        self._replay_hit = 0
        self._recapture = 0
        self._evict = 0
        self._miss_reasons: Dict[str, int] = {r: 0 for r in MISS_REASONS}

    @property
    def cache(self) -> "torch.rpu.GraphCache":
        return self._cache

    @property
    def planner(self) -> SignaturePlanner:
        return self._planner

    def evict_total(self) -> int:
        return self._evict

    def admit_for_call(self, plan: GraphPlan) -> Any:
        """Return the backend RpuKernelGraph handle for this plan.

        Backend `g.begin(sig)` remains the authority for RECORDING vs
        REPLAYING. This method only routes to a cache entry and evicts older
        entries before admitting a new one when the frontend LRU is full.
        """
        pool = self._pool
        key = SignaturePlanner.signature_key(plan.signature)

        if key in pool:
            pool.move_to_end(key)
            entry = pool[key]
            entry.plan = plan
            return self._cache.get_or_create(_sig_impl(plan.signature))

        if not pool:
            miss_reason = MISS_NO_ENTRY
        else:
            recent_sig = next(reversed(pool.values())).signature
            miss_reason = (
                SignaturePlanner.attribute_miss_reason(
                    plan.signature, recent_sig)
                or MISS_NO_ENTRY
            )
        self._miss_reasons[miss_reason] += 1

        while len(pool) >= self._max_entries:
            old_key, old_entry = pool.popitem(last=False)
            try:
                self._cache.evict(_sig_impl(old_entry.signature))
            except Exception:
                pool[old_key] = old_entry
                pool.move_to_end(old_key)
                break
            else:
                self._evict += 1

        pool[key] = _PoolEntry(signature=plan.signature, plan=plan)
        return self._cache.get_or_create(_sig_impl(plan.signature))

    def touch_after_call(self, plan: GraphPlan, in_scope_state: int) -> None:
        """Update frontend counters after `g.begin / call / g.end`."""
        pool = self._pool
        key = SignaturePlanner.signature_key(plan.signature)
        entry = pool.get(key)
        if entry is not None:
            entry.last_state = in_scope_state
        if in_scope_state == STATE_REPLAYING:
            self._replay_hit += 1
            if entry is not None:
                entry.replay_count += 1
            self._cache.touch_entry(_sig_impl(plan.signature), replay=True)
        elif in_scope_state == STATE_RECORDING:
            previously_replaying = (
                entry is not None
                and entry.replay_count > 0
            )
            if previously_replaying and \
                    plan.expected_admission == ADMISSION_REPLAY:
                self._recapture += 1
                if entry is not None:
                    entry.recapture_count += 1
            self._cache.touch_entry(_sig_impl(plan.signature), replay=False)

    def stats(self) -> Dict[str, Any]:
        """Return registry-level counters and miss-reason diagnostics."""
        miss_reason_total = sum(self._miss_reasons.values())
        return {
            "replay_hit_total": self._replay_hit,
            "recapture_total": self._recapture,
            "evict_total": self.evict_total(),
            "pool_size": len(self._pool),
            "max_entries": self._max_entries,
            "miss_reasons": dict(self._miss_reasons),
            "miss_reason_total": miss_reason_total,
        }

    def clear(self) -> None:
        """Drop frontend entries without mutating the backend GraphCache."""
        self._pool.clear()
        self._replay_hit = 0
        self._recapture = 0
        self._evict = 0
        self._miss_reasons = {r: 0 for r in MISS_REASONS}


__all__ = [
    "ADMISSION_KINDS",
    "ADMISSION_RECORD",
    "ADMISSION_RECAPTURE",
    "ADMISSION_REPLAY",
    "EntryRegistry",
    "GraphPlan",
    "LIFECYCLE_EVICT",
    "LIFECYCLE_KINDS",
    "LIFECYCLE_RECAPTURE",
    "LIFECYCLE_RETAIN",
    "MISS_BRANCH_KEY",
    "MISS_DTYPE_LAYOUT",
    "MISS_GM_ID",
    "MISS_NO_ENTRY",
    "MISS_REASONS",
    "MISS_SCALAR_FLAGS",
    "MISS_SEGMENT_KEY",
    "MISS_SHAPE_BUCKET",
    "MISS_TENSOR_STRUCT",
    "SignaturePlanner",
    "STATE_BUILT",
    "STATE_PASSTHROUGH",
    "STATE_RECORDING",
    "STATE_REPLAYING",
    "WORKLOAD_DECODE",
    "WORKLOAD_GENERIC",
    "WORKLOAD_KINDS",
    "WORKLOAD_PREFILL",
    "build_plan_for_call",
]
