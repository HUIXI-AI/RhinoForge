"""Small, deterministic host facade for execution-plan selection.

The native FusedModelBase planner remains the safety authority.  This module
only owns the outer, finite ``execution_len x (input,qkv,compute)`` search so
callers can use one auto/config path and get an auditable result.  A caller
supplies the stage-domain/feasibility oracle; this module never guesses model
semantics from tensor shapes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import OrderedDict
from functools import cached_property
from hashlib import sha256
from itertools import islice
import json
from numbers import Integral
import re
from typing import Callable, Iterable, Mapping, Sequence

if __package__:
    from rpu_backend.runtime import RPUBackendError
else:
    # Offline receipt verification consumes this same planner without importing
    # the device-owning package. Only its stdlib error base needs a leaf import.
    from pathlib import Path
    _error_path = Path(__file__).with_name("_errors.py")
    _read_source = getattr(globals().get("__loader__"), "read", Path.read_bytes)
    _errors = {"__name__": "_rpu_offline_errors", "__file__": str(_error_path)}
    exec(compile(_read_source(_error_path), str(_error_path), "exec"), _errors)
    RPUBackendError = _errors["RPUBackendError"]

MAX_A6_PAIRS = 4096
MAX_EXECUTION_LENGTH_CANDIDATES = 256
MAX_DIAG_RECORDS = 4096
_MAX_NATIVE_INT64 = (1 << 63) - 1


class PreparedExecutionPlans:
    """Owner-local, bounded preparation shared by retained and one-shot graphs.

    Keys contain caller-declared semantics, never the identity of a resolver
    closure. A failed preparation publishes nothing. READY is lookup-only,
    including on a plan miss before the physical Graph is consulted.
    """

    def __init__(self, max_entries=32):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("plan cache capacity must be a positive integer")
        self.max_entries = max_entries
        self._plans = OrderedDict()
        self._ready = False
        self._last_input = None
        self._last_result = None

    def prepare(self, key, factory):
        # The last hit is already the newest LRU entry. Repeated immutable
        # requests need neither a fresh recursive type-tag tree nor its hash.
        # Type-aware comparison still rejects Python's True == 1 alias.
        if self._last_input is not None and self._same_immutable_key(key, self._last_input):
            return self._last_result
        # Type tags prevent bool/int and other Python equality aliases from
        # reusing a plan for a malformed or semantically different request.
        typed_key = self._key(key)
        if typed_key in self._plans:
            self._plans.move_to_end(typed_key)
            result = self._plans[typed_key]
        else:
            if self._ready:
                raise RuntimeError("execution plan READY miss; call begin_warmup() before preparing a new signature")
            result = factory()
            self._plans[typed_key] = result
            if len(self._plans) > self.max_entries:
                self._plans.popitem(last=False)
        # Mutable lists (including nested ones) must be revalidated next time.
        self._last_input = key if self._immutable_key(key) else None
        self._last_result = result
        return result

    @classmethod
    def _immutable_key(cls, value):
        return (value is None or type(value) in (bool, int, str) or
                (type(value) is tuple and all(cls._immutable_key(item) for item in value)))

    @classmethod
    def _same_immutable_key(cls, value, previous):
        if value is previous:
            return True
        if type(value) is not type(previous):
            return False
        if type(value) is tuple:
            if len(value) != len(previous):
                return False
            for item, old in zip(value, previous):
                if item is old:
                    continue
                if type(item) is not type(old):
                    return False
                if type(item) is tuple:
                    if not cls._same_immutable_key(item, old):
                        return False
                elif item != old:
                    return False
            return True
        return value == previous

    @classmethod
    def _key(cls, value):
        if value is None or type(value) in (bool, int, str):
            return (type(value).__name__, value)
        if isinstance(value, (tuple, list)):
            return ("sequence", tuple(cls._key(item) for item in value))
        raise TypeError("plan signature must contain only immutable scalar semantics")

    def begin_warmup(self):
        self._ready = False

    def freeze(self):
        self._ready = True

    def clear(self):
        self._plans.clear()
        self._last_input = self._last_result = None

    def __len__(self):
        return len(self._plans)

REQUEST_AUTO = "AUTO"
REQUEST_EXACT = "EXACT"
REQUEST_MODES = (REQUEST_AUTO, REQUEST_EXACT)

GRAPH_RETAINED_CACHE = "RETAINED_CACHE"
GRAPH_BOUNDED_ONESHOT = "BOUNDED_ONESHOT"
GRAPH_NATIVE_COMPOSITE1 = "NATIVE_COMPOSITE1"
GRAPH_COMPOSITE_CHILD = "COMPOSITE_CHILD"
GRAPH_MODES = (
    GRAPH_RETAINED_CACHE,
    GRAPH_BOUNDED_ONESHOT,
    GRAPH_NATIVE_COMPOSITE1,
    GRAPH_COMPOSITE_CHILD,
)


@dataclass(frozen=True)
class PlannerCostScope:
    """Internal immutable profile and dependency digests, not caller labels.

    The owner must bind geometry, precision, capability and physical site in
    profile_identity. dependency_identity is the existing verified dependency
    manifest SHA-256 (SDK/operator artifacts); runtime_identity is the existing
    runtime_payload_manifest_sha256. An absent scope cannot inherit a certificate.
    """

    profile_identity: str
    dependency_identity: str
    runtime_identity: str

    def __post_init__(self) -> None:
        for value in (self.profile_identity, self.dependency_identity, self.runtime_identity):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("planner cost identities must be SHA-256 digests")


@dataclass(frozen=True)
class PlannerCostCertificate:
    scope: PlannerCostScope
    domain_digest: str
    # Exact native-candidate digest -> same-session median device time (ns).
    candidate_costs: tuple[tuple[str, int], ...]
    receipt_sha256: str
    sealed: bool = False


# P7 data is injected as an immutable owner-local tuple after artifact
# verification, never compiled into the runtime payload whose hash it binds.
# There is deliberately no process-global catalog or executor/config parser.


class PlannerRejectError(RPUBackendError, RuntimeError):
    """A plan rejection, retaining the legacy ``RuntimeError`` catch contract."""

    def __init__(
        self,
        code: str,
        *,
        stage: str = "CONTROL",
        requested: int | None = None,
        limit: int | None = None,
        resolved: int | None = None,
        clamped: bool = False,
        detail: str = "",
    ) -> None:
        self.code = str(code)
        self.stage = str(stage)
        self.requested = requested
        self.limit = limit
        self.resolved = resolved
        self.clamped = bool(clamped)
        message = f"planner rejected: code={self.code} stage={self.stage}"
        if detail:
            message += f" ({detail})"
        super().__init__(message)


def native_chunk_reject(
    error: RuntimeError, *, stage: str, requested: int,
) -> PlannerRejectError | None:
    """Translate only the native resolver's stable no-candidate contract.

    PyTorch exposes every ``TORCH_CHECK`` as ``RuntimeError``.  Handle, device,
    initialization, and programming failures must therefore propagate; only
    the versioned native rejection prefix is translated.
    """
    message = str(error)
    match = re.search(r"(?:^|\s)RPU_PLANNER_REJECT:([A-Z_]+):", message)
    if match is None:
        return None
    return PlannerRejectError(
        match.group(1), stage=stage, requested=requested, detail=message,
    )


@dataclass(frozen=True)
class PlannerRequest:
    """Normalized caller/config input for one A6 search."""

    mode: str = REQUEST_AUTO
    chunk_size: int | None = None
    padding_rows: int | None = None
    padding_budget: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "PlannerRequest":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError(f"planner request must be a mapping, got {type(value).__name__}")
        unknown = set(value) - {"mode", "chunk_size", "padding_rows", "padding_budget"}
        if unknown:
            raise ValueError(f"planner request has unknown field(s): {sorted(map(str, unknown))}")

        mode_explicit = "mode" in value
        raw_mode = value.get("mode", REQUEST_AUTO)
        if not isinstance(raw_mode, str):
            raise ValueError("planner request 'mode' must be AUTO or EXACT")
        mode = raw_mode.upper()
        if mode not in REQUEST_MODES:
            raise ValueError(f"planner request 'mode' must be one of {REQUEST_MODES}, got {raw_mode!r}")

        raw_chunk = value.get("chunk_size")
        chunk: int | None
        if raw_chunk is None or raw_chunk == "auto":
            chunk = None
        elif isinstance(raw_chunk, bool) or not isinstance(raw_chunk, Integral):
            raise ValueError("planner request 'chunk_size' must be 'auto' or a positive multiple of 16")
        else:
            chunk = int(raw_chunk)
            if chunk <= 0 or chunk % 16:
                raise ValueError("planner request 'chunk_size' must be 'auto' or a positive multiple of 16")
            if not mode_explicit:
                mode = REQUEST_EXACT
            elif mode == REQUEST_AUTO:
                raise ValueError(
                    "AUTO planner request conflicts with an exact chunk_size"
                )

        raw_padding = value.get("padding_rows")
        padding: int | None
        if raw_padding is None or raw_padding == "auto":
            padding = None
        elif isinstance(raw_padding, bool) or not isinstance(raw_padding, Integral):
            raise ValueError("planner request 'padding_rows' must be 'auto' or a non-negative integer")
        else:
            padding = int(raw_padding)
            if padding < 0:
                raise ValueError("planner request 'padding_rows' must be 'auto' or a non-negative integer")

        raw_budget = value.get("padding_budget", 0)
        if isinstance(raw_budget, bool) or not isinstance(raw_budget, Integral) or int(raw_budget) < 0:
            raise ValueError("planner request 'padding_budget' must be a non-negative integer")
        budget = int(raw_budget)
        if padding is not None and "padding_budget" in value:
            raise ValueError("planner request 'padding_rows' conflicts with 'padding_budget'")
        if mode == REQUEST_EXACT and chunk is None:
            raise ValueError("EXACT planner request requires chunk_size")
        return cls(mode, chunk, padding, budget)

    def as_dict(self) -> dict[str, object]:
        """Return the detached, canonical request fields used by receipts."""
        return {
            "mode": self.mode,
            "chunk_size": self.chunk_size,
            "padding_rows": self.padding_rows,
            "padding_budget": self.padding_budget,
        }


@dataclass(frozen=True)
class StageTuple:
    input_chunk: int
    qkv_chunk: int
    compute_chunk: int
    physical_metadata: tuple[tuple[str, int], ...] = ()
    physical_descriptor: tuple[int, ...] = ()

    @classmethod
    def from_value(cls, value: object) -> "StageTuple":
        if isinstance(value, cls):
            values = (value.input_chunk, value.qkv_chunk, value.compute_chunk)
            raw_metadata = value.physical_metadata
            raw_descriptor = value.physical_descriptor
        else:
            try:
                values = tuple(value)  # type: ignore[arg-type]
            except TypeError as exc:
                raise ValueError("stage tuple must contain input/qkv/compute chunks") from exc
            if len(values) != 3:
                raise ValueError("stage tuple must contain input/qkv/compute chunks")
            raw_metadata = ()
            raw_descriptor = ()
        # Do not use ``int(v)`` here: silently truncating 64.5 (or accepting
        # 64.0) would make the digest describe a plan different from the
        # caller's request.
        chunks = tuple(
            _checked_int("stage tuple chunk", item, minimum=1) for item in values
        )
        if chunks[1] % 16 or chunks[2] % 16:
            raise ValueError("QKV/compute stage chunks must be positive multiples of 16")
        metadata: list[tuple[str, int]] = []
        for item in raw_metadata:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("physical metadata must contain (name, integer) pairs")
            name, raw_value = item
            if not isinstance(name, str) or not name:
                raise ValueError("physical metadata names must be non-empty strings")
            if isinstance(raw_value, bool) or not isinstance(raw_value, Integral):
                raise ValueError("physical metadata values must be integers")
            metadata.append((name, int(raw_value)))
        metadata.sort()
        if len({name for name, _ in metadata}) != len(metadata):
            raise ValueError("physical metadata names must be unique")
        if any(
            type(item) is not int
            and (isinstance(item, bool) or not isinstance(item, Integral))
            for item in raw_descriptor
        ):
            raise ValueError("physical descriptor words must be integers")
        descriptor = tuple(int(item) for item in raw_descriptor)
        return cls(*chunks, tuple(metadata), descriptor)

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.input_chunk, self.qkv_chunk, self.compute_chunk)

    def identity(self) -> tuple[object, ...]:
        return (*self.as_tuple(), self.physical_metadata,
                self.physical_descriptor)


@dataclass(frozen=True)
class PlannerCalibrationSelection:
    """Private P7 selection of an admitted row, never a user configuration."""

    scope: PlannerCostScope
    domain_digest: str
    candidate_digest: str
    native_route: tuple[int, int, int] | None = None
    domain_scoped: bool = False

    def __post_init__(self):
        if not isinstance(self.scope, PlannerCostScope) or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in (self.domain_digest, self.candidate_digest)
        ):
            raise ValueError("calibration requires an actual scope and full domain/candidate SHA-256")
        if type(self.domain_scoped) is not bool:
            raise ValueError("calibration domain_scoped must be bool")
        if self.native_route is not None and (
            not isinstance(self.native_route, tuple) or len(self.native_route) != 3
            or any(type(value) is not int or value < minimum or value >= 2**63
                   for value, minimum in zip(self.native_route, (1, 0, 1)))
        ):
            raise ValueError("calibration native route requires (site, invocation, route)")


@dataclass(frozen=True)
class A6Candidate:
    execution_len: int
    padding_rows: int
    stage_tuple: StageTuple
    num_chunks: tuple[int, int, int]
    tail_deficit: tuple[int, int, int]
    score: tuple[int, ...]
    feasible: bool
    reject_reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "execution_len": self.execution_len,
            "padding_rows": self.padding_rows,
            "input_chunk": self.stage_tuple.input_chunk,
            "qkv_chunk": self.stage_tuple.qkv_chunk,
            "compute_chunk": self.stage_tuple.compute_chunk,
            "physical_metadata": dict(self.stage_tuple.physical_metadata),
            "descriptor_words": len(self.stage_tuple.physical_descriptor),
            "num_chunks": self.num_chunks,
            "tail_deficit": self.tail_deficit,
            "score": self.score,
            "feasible": self.feasible,
            "reject_reason": self.reject_reason,
        }


@dataclass(frozen=True)
class A6PlanResult:
    status: str
    request: PlannerRequest
    selected: A6Candidate | None
    candidates: tuple[A6Candidate, ...]
    candidate_count: int
    feasible_count: int
    rejected_count: int
    search_complete: bool
    optimality: str
    graph_mode: str
    queue_owner_id: int
    lease_owner_id: int
    domain_digest: str
    physical_plan_digest: str
    plan_digest: str
    truncated: bool = False
    clamped: bool = False
    selection_scope: str = "FULL_STAGE_DOMAIN"
    domain_rejections: tuple[tuple[int, str], ...] = ()
    cost_certificate: PlannerCostCertificate | None = None
    cost_scope: PlannerCostScope | None = None
    cost_domain_digest: str = ""
    selected_cost_candidate_digest: str = ""
    # Original, unranked feasible rows; never reconstruct them by stripping
    # cost metadata from the winner. FINAL derives these again from raw input.
    cost_feasible_pairs: tuple[tuple[int, StageTuple], ...] = ()
    # Native route catalogs cover the physically feasible raw owner domain;
    # an external exact chunk filters outer choices, not that parent catalog.
    native_cost_feasible_pairs: tuple[tuple[int, StageTuple], ...] = ()

    @cached_property
    def _receipt_template(self) -> dict[str, object]:
        # Private static fields of this immutable plan; public receipts copy
        # every mutable level below before returning to adapters or callers.
        request = self.request.as_dict()
        selected = self.selected.as_dict() if self.selected else None
        if selected is not None:
            # Keep the dispatched winner in the receipt. Candidate diagnostics
            # intentionally retain only descriptor_words, so search logs stay
            # bounded even when the native domain is large.
            selected["physical_descriptor"] = (
                self.selected.stage_tuple.physical_descriptor
            )
        result: dict[str, object] = {
            "status": self.status,
            "request_mode": request["mode"],
            "request": request,
            "requested_chunk_size": request["chunk_size"],
            "requested_padding_rows": request["padding_rows"],
            "padding_budget": request["padding_budget"],
            "selected": selected,
            "resolved_chunk_size": (
                selected["compute_chunk"] if selected is not None else None
            ),
            "resolved_padding_rows": (
                selected["padding_rows"] if selected is not None else None
            ),
            "candidate_count": self.candidate_count,
            "feasible_count": self.feasible_count,
            "rejected_count": self.rejected_count,
            "search_complete": self.search_complete,
            "optimality": self.optimality,
            "selected_cost_candidate_digest": self.selected_cost_candidate_digest or None,
            "selection_scope": self.selection_scope,
            "graph_mode": self.graph_mode,
            "queue_owner_id": self.queue_owner_id,
            "lease_owner_id": self.lease_owner_id,
            "owner_identity_status": (
                "RESOLVED"
                if self.queue_owner_id > 0
                and (
                    self.graph_mode != GRAPH_NATIVE_COMPOSITE1
                    or self.lease_owner_id > 0
                )
                else "UNRESOLVED"
            ),
            "domain_digest": self.domain_digest,
            "physical_plan_digest": self.physical_plan_digest,
            "plan_digest": self.plan_digest,
            "truncated": self.truncated,
            "clamped": self.clamped,
            "hardware_cost": {
                "status": (
                    "SEALED_EXACT_DOMAIN" if self.cost_certificate is not None
                    else "NO_MATCHING_SEALED_CERTIFICATE" if self.cost_scope is not None
                    else "PROFILE_IDENTITY_UNBOUND"
                ),
                "profile_identity": (
                    self.cost_scope.profile_identity if self.cost_scope else None
                ),
                "dependency_identity": (
                    self.cost_scope.dependency_identity if self.cost_scope else None
                ),
                "runtime_identity": (
                    self.cost_scope.runtime_identity if self.cost_scope else None
                ),
                "domain_digest": self.cost_domain_digest or None,
                "receipt_sha256": (
                    self.cost_certificate.receipt_sha256
                    if self.cost_certificate else None
                ),
                "metric": "same_session_median_device_ns",
            },
        }
        return result

    def as_dict(self, *, include_candidates: bool = True) -> dict[str, object]:
        result = self._receipt_template.copy()
        result["request"] = result["request"].copy()
        result["hardware_cost"] = result["hardware_cost"].copy()
        if result["selected"] is not None:
            selected = result["selected"].copy()
            selected["physical_metadata"] = selected["physical_metadata"].copy()
            result["selected"] = selected
        if include_candidates:
            result["candidates"] = tuple(c.as_dict() for c in self.candidates)
            result["domain_rejections"] = self.domain_rejections
        return result

    @cached_property
    def _cached_graph_key_words(self) -> tuple[int, ...]:
        return tuple(
            int(self.physical_plan_digest[offset:offset + 8], 16)
            for offset in range(0, len(self.physical_plan_digest), 8)
        )

    def graph_key_words(self) -> tuple[int, ...]:
        """Return all digest words, decoded once for this immutable result."""
        return self._cached_graph_key_words


def _digest(value: object, domain: str) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return sha256(domain.encode() + b"\0" + payload).hexdigest()


def planner_cost_domain(
    domain: Mapping[int, Sequence[StageTuple]], *, scope: PlannerCostScope,
    logical_len: int, position: int, graph_mode: str,
    feasible_pairs=None,
) -> tuple[str, dict[tuple[int, StageTuple], str]]:
    """Identify the exact domain exported for the existing P7 HWPerf sweep.

    Full descriptors include spans, padding, selectors and invocation/site
    identities. Scope additionally binds model geometry/precision; equal
    request_id strings or equal chunk triples cannot authorize inheritance.
    """
    if not isinstance(scope, PlannerCostScope):
        raise TypeError("cost_scope must be a PlannerCostScope")
    keys = {
        (int(length), row): _digest(
            # Cache generations retire Graphs, not measurements of an unchanged
            # physical candidate. Keep them in row/Graph identity, but not cost
            # identity; all other metadata and the full descriptor stay bound.
            (int(length), replace(row, physical_metadata=tuple(
                (name, value) for name, value in row.physical_metadata
                if name not in {"execution_generation", "component_generation"}
            )).identity()), "RPU-COST-CANDIDATE-V1",
        )
        for length, rows in domain.items()
        for row in (StageTuple.from_value(item) for item in rows)
    }
    raw_keys = sorted(set(keys.values()))
    if feasible_pairs is not None:
        feasible_pairs = frozenset(feasible_pairs)
        if not feasible_pairs.issubset(keys):
            raise ValueError("feasible cost candidates must belong to the actual raw domain")
        keys = {pair: key for pair, key in keys.items() if pair in feasible_pairs}
    digest = _digest({
        "profile_identity": scope.profile_identity,
        "dependency_identity": scope.dependency_identity,
        "runtime_identity": scope.runtime_identity,
        "logical_len": logical_len,
        "position": position,
        "graph_mode": graph_mode,
        "raw_candidates": raw_keys,
        "candidates": sorted(set(keys.values())),
    }, "RPU-COST-DOMAIN-V1")
    return digest, keys


def _apply_cost_certificate(
    domain: Mapping[int, Sequence[StageTuple]], *, scope: PlannerCostScope | None,
    logical_len: int, position: int, graph_mode: str,
    certificates: tuple[PlannerCostCertificate, ...] = (),
    feasible_pairs=None,
) -> tuple[Mapping[int, Sequence[StageTuple]], PlannerCostCertificate | None, str,
           dict[tuple[int, StageTuple], str]]:
    if not isinstance(certificates, tuple) or any(
        not isinstance(item, PlannerCostCertificate) or
        not isinstance(item.candidate_costs, tuple) or
        any(not isinstance(row, tuple) for row in item.candidate_costs)
        for item in certificates
    ):
        raise TypeError("planner costs require an immutable certificate tuple")
    if scope is None:
        if certificates:
            raise ValueError("planner cost certificates require a bound cost_scope")
        return domain, None, "", {}
    domain_digest, keys = planner_cost_domain(
        domain, scope=scope, logical_len=logical_len,
        position=position, graph_mode=graph_mode,
        feasible_pairs=feasible_pairs,
    )
    matches = [
        item for item in certificates
        if item.sealed is True and item.scope == scope and item.domain_digest == domain_digest
    ]
    if not matches:
        return domain, None, domain_digest, keys
    if len(matches) != 1:
        raise ValueError("planner cost domain has duplicate sealed certificates")
    certificate = matches[0]
    if re.fullmatch(r"[0-9a-f]{64}", certificate.receipt_sha256) is None:
        raise ValueError("planner cost certificate requires a receipt SHA-256")
    costs = dict(certificate.candidate_costs)
    if (len(costs) != len(certificate.candidate_costs) or
            set(costs) != set(keys.values())):
        raise ValueError("planner cost certificate must cover the exact candidate domain")
    for cost in costs.values():
        _checked_int("hardware candidate cost", cost, minimum=1)
    if any(not row.physical_descriptor or
           dict(row.physical_metadata).get("manifest_state") != 1
           for _length, row in keys):
        raise ValueError("hardware costs require COMPLETE native descriptors")
    # Preserve the full receipt in the result; the positive int is only the
    # existing native-metadata selection marker, never identity authority.
    marker = int(certificate.receipt_sha256[:15], 16) or 1
    ranked, ranked_keys = {}, {}
    for length, rows in domain.items():
        rows = _normalise_domain(rows, execution_len=length)
        updated_rows = []
        for row in rows:
            if (length, row) not in keys:
                updated_rows.append(row)
                continue
            updated = replace(row, physical_metadata=tuple(sorted({
                **dict(row.physical_metadata),
                "hardware_cost_certificate": marker,
                "hardware_cost_score": costs[keys[(length, row)]],
                "selector_state": 2,
            }.items())))
            key = (length, updated)
            digest = keys[(length, row)]
            if key in ranked_keys and ranked_keys[key] != digest:
                raise ValueError("cost metadata update merged distinct physical candidates")
            ranked_keys[key] = digest
            updated_rows.append(updated)
        ranked[length] = tuple(sorted(updated_rows, key=lambda row: row.identity()))
    return ranked, certificate, domain_digest, ranked_keys


def _checked_int(name: str, value: int, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _execution_lengths(
    logical_len: int,
    *,
    alignment: int,
    physical_limit: int | None,
    request: PlannerRequest,
    max_lengths: int,
) -> tuple[tuple[int, ...], bool]:
    base = ((logical_len + alignment - 1) // alignment) * alignment
    if base > _MAX_NATIVE_INT64:
        raise PlannerRejectError(
            "CAPABILITY", requested=base, limit=_MAX_NATIVE_INT64,
            detail="aligned execution length exceeds the native int64 domain",
        )
    if request.padding_rows is not None:
        execution_len = logical_len + request.padding_rows
        if execution_len > _MAX_NATIVE_INT64:
            raise PlannerRejectError(
                "CAPABILITY", requested=execution_len, limit=_MAX_NATIVE_INT64,
                detail="exact padded execution length exceeds the native int64 domain",
            )
        values = (
            () if physical_limit is not None and execution_len > physical_limit
            else (execution_len,)
        )
    else:
        # Alignment padding is mandatory; the budget controls only the
        # optional search window after the smallest aligned execution length.
        upper = base + request.padding_budget
        if physical_limit is not None:
            upper = min(upper, physical_limit)
        if upper > _MAX_NATIVE_INT64:
            raise PlannerRejectError(
                "CAPABILITY", requested=upper, limit=_MAX_NATIVE_INT64,
                detail="padded execution-length domain exceeds native int64",
            )
        count = 0 if upper < base else (upper - base) // alignment + 1
        if count > max_lengths:
            raise PlannerRejectError(
                "SEARCH_LIMIT", requested=count, limit=max_lengths,
                detail="execution-length candidate domain exceeds limit",
            )
        values = tuple(base + index * alignment for index in range(count))
    if any(v % alignment for v in values):
        raise PlannerRejectError(
            "ALIGNMENT", requested=logical_len, detail=(
                "execution-length candidate is not aligned to the requested "
                f"boundary {alignment}"
            ),
        )
    if len(values) > max_lengths:
        raise PlannerRejectError(
            "SEARCH_LIMIT", requested=len(values), limit=max_lengths,
            detail="execution-length candidate domain exceeds limit",
        )
    return values, False


def _physical_manifest_fingerprint(values: Iterable[int]) -> int:
    """Mirror the native FNV-1a/u64 physical-manifest codec."""
    mask = (1 << 64) - 1
    value_hash = 1469598103934665603

    def mix(raw_value: int) -> None:
        nonlocal value_hash
        encoded = raw_value & mask
        for shift in range(0, 64, 8):
            value_hash ^= (encoded >> shift) & 0xFF
            value_hash = (value_hash * 1099511628211) & mask

    mix(0x464D425F50485932)  # "FMB_PHY2"
    for value in values:
        mix(value)
    return value_hash or 1


_ROPE_TABLE_DDR_FLAG = 1 << 56
_ROPE_TABLE_SPM_FLAG = 1 << 57


def _decode_native_stage_domain(
    values: Iterable[object], *, execution_len: int, logical_len: int,
    position: int, graph_mode: str,
) -> tuple[StageTuple, ...]:
    """Decode domain-v1 framing with native candidate descriptor v1/v2/v3."""
    raw = tuple(values)
    # Native descriptors contain Python ints. Avoid an ABC lookup per word,
    # while retaining Integral subclasses and rejecting bool/float inputs.
    if any(
        type(item) is not int
        and (isinstance(item, bool) or not isinstance(item, Integral))
        for item in raw
    ):
        raise ValueError("native stage-domain wire must contain only integers")
    words = tuple(int(item) for item in raw)
    if (
        len(words) < 2
        or words[0] != 1
        or not 0 <= words[1] <= MAX_A6_PAIRS
    ):
        raise ValueError("native stage-domain wire has an invalid v1 header")

    cursor = 2
    rows: list[StageTuple] = []
    for _ in range(words[1]):
        if cursor >= len(words):
            raise ValueError("native stage-domain wire is truncated")
        record_words = words[cursor]
        cursor += 1
        if record_words < 15 or record_words > 65536 or cursor + record_words > len(words):
            raise ValueError("native stage-domain candidate length is invalid")
        descriptor = words[cursor:cursor + record_words]
        cursor += record_words
        descriptor_version = descriptor[0]
        if descriptor_version not in (1, 2, 3) or descriptor[1] != record_words:
            raise ValueError(
                "native stage-domain candidate has an invalid v1/v2/v3 header"
            )

        capacities = descriptor[2:5]
        if (
            any(value <= 0 for value in capacities)
            or capacities[1] % 16
            or capacities[2] % 16
        ):
            raise ValueError("native stage-domain candidate has invalid capacities")
        if (
            not 0 <= descriptor[5] <= 0xFFFFFFFF
            or not 0 <= descriptor[6] <= 0xFFFFFFFF
            or descriptor[7] not in (0, 1)
            or any(value not in (0, 1, 2) for value in descriptor[8:11])
        ):
            raise ValueError("native stage-domain candidate has invalid policy metadata")

        counts = descriptor[11:14]
        span_count = descriptor[14]
        if any(value <= 0 for value in (*counts, span_count)):
            raise ValueError("native stage-domain candidate has an invalid schedule count")
        stage_words = 15 + 4 * sum(counts) + 3 * span_count
        if (
            (descriptor_version == 1 and stage_words != record_words)
            or (descriptor_version in (2, 3) and stage_words + 11 > record_words)
        ):
            raise ValueError(
                "native stage-domain candidate payload size is inconsistent"
            )

        item_cursor = 15
        tail_deficits: list[int] = []
        tail_offsets: list[int] = []
        for capacity, count in zip(capacities, counts):
            expected_offset = 0
            for ordinal in range(count):
                index = descriptor[item_cursor]
                offset = descriptor[item_cursor + 1]
                length = descriptor[item_cursor + 2]
                kv_seq_len = descriptor[item_cursor + 3]
                item_cursor += 4
                if (
                    index != ordinal
                    or offset != expected_offset
                    or length <= 0
                    or length > capacity
                    or (ordinal + 1 < count and length != capacity)
                    or kv_seq_len != position + offset + length
                ):
                    raise ValueError("native stage-domain candidate has an invalid chunk schedule")
                expected_offset += length
            if expected_offset != execution_len:
                raise ValueError("native stage-domain stage does not cover execution_len")
            tail_offsets.append(expected_offset - length)
            tail_deficits.append(capacity - length)

        expected_span_offset = 0
        for _ in range(span_count):
            offset = descriptor[item_cursor]
            length = descriptor[item_cursor + 1]
            group_id = descriptor[item_cursor + 2]
            item_cursor += 3
            if (
                offset != expected_span_offset
                or length <= 0
                or (span_count > 1 and group_id < 0)
            ):
                raise ValueError("native stage-domain candidate has an invalid semantic span")
            expected_span_offset += length
        if expected_span_offset != execution_len:
            raise ValueError("native stage-domain semantic spans do not cover execution_len")

        physical_metadata: list[tuple[str, int]] = []
        if descriptor_version == 1:
            if item_cursor != record_words:
                raise ValueError(
                    "native stage-domain candidate payload size is inconsistent"
                )
        else:
            # Cython's parallel-assignment pass crashes on a dynamic slice
            # unpack here after the closed-source normalizer compacts it.
            manifest_state = descriptor[item_cursor]
            logical_length = descriptor[item_cursor + 1]
            physical_length = descriptor[item_cursor + 2]
            execution_padding_rows = descriptor[item_cursor + 3]
            kv_logical_length = descriptor[item_cursor + 4]
            kv_insert_physical_rows = descriptor[item_cursor + 5]
            graph_lifecycle = descriptor[item_cursor + 6]
            linear_acc_policy = descriptor[item_cursor + 7]
            route_count = descriptor[item_cursor + 8]
            manifest_fingerprint_hi = descriptor[item_cursor + 9]
            manifest_fingerprint_lo = descriptor[item_cursor + 10]
            item_cursor += 11
            if route_count < 0:
                raise ValueError("native physical manifest has an invalid route count")
            if (
                not 0 <= manifest_fingerprint_hi <= 0xFFFFFFFF
                or not 0 <= manifest_fingerprint_lo <= 0xFFFFFFFF
            ):
                raise ValueError(
                    "native physical manifest has invalid fingerprint words"
                )

            routes: list[
                tuple[int, int, int, int, tuple[int, ...], int]
            ] = []
            previous_identity = (-1, -1, -1)
            for _ in range(route_count):
                route_header_words = 6 if descriptor_version == 3 else 5
                if item_cursor + route_header_words > record_words:
                    raise ValueError("native physical manifest route is truncated")
                family = descriptor[item_cursor]
                site_id = descriptor[item_cursor + 1]
                selector = descriptor[item_cursor + 2]
                flags = descriptor[item_cursor + 3]
                argument_count = descriptor[item_cursor + 4]
                invocation = (
                    descriptor[item_cursor + 5]
                    if descriptor_version == 3
                    else 0
                )
                item_cursor += route_header_words
                if argument_count < 0 or item_cursor + argument_count > record_words:
                    raise ValueError(
                        "native physical manifest route arguments are truncated"
                    )
                arguments = tuple(
                    descriptor[item_cursor:item_cursor + argument_count]
                )
                item_cursor += argument_count
                identity = (family, site_id, invocation)
                if (
                    family not in range(1, 11)
                    or site_id <= 0
                    or selector <= 0
                    or flags < 0
                    or invocation < 0
                    or any(argument < 0 for argument in arguments)
                    or identity <= previous_identity
                ):
                    raise ValueError(
                        "native physical manifest route is invalid or non-canonical"
                    )
                routes.append((
                    family, site_id, selector, flags, arguments, invocation,
                ))
                previous_identity = identity
            if item_cursor != record_words:
                raise ValueError(
                    "native physical manifest has trailing words"
                )

            attention_ddr_site_count = len({
                site
                for family, site, selector, _flags, _args, _inv in routes
                if family == 1 and selector == 1
            })
            raw_attention_site_count = len({
                site
                for family, site, selector, _flags, _args, _inv in routes
                if family == 1 and selector == 2
            })
            rope_table_residency = 0
            for family, _site, _selector, flags, _args, _inv in routes:
                if family != 4:
                    continue
                residency_flags = flags & (
                    _ROPE_TABLE_DDR_FLAG | _ROPE_TABLE_SPM_FLAG
                )
                if residency_flags == 0:
                    continue
                if residency_flags == (
                    _ROPE_TABLE_DDR_FLAG | _ROPE_TABLE_SPM_FLAG
                ):
                    raise ValueError(
                        "native ROPE route has conflicting table residency"
                    )
                route_residency = (
                    2 if residency_flags == _ROPE_TABLE_SPM_FLAG else 1
                )
                if rope_table_residency not in (0, route_residency):
                    raise ValueError(
                        "native physical manifest mixes ROPE table residencies"
                    )
                rope_table_residency = route_residency
            # This is a deterministic capability rank, not a measured latency
            # certificate: prefer fewer DDR attention sites, then an applicable
            # SPM RoPE table route.  The manifest fingerprint remains only the
            # final total-order key.
            rope_residency_rank = 1 if rope_table_residency == 1 else 0

            if manifest_state == 0:
                if any((logical_length, physical_length,
                        execution_padding_rows, kv_logical_length,
                        kv_insert_physical_rows, graph_lifecycle,
                        linear_acc_policy, route_count)):
                    raise ValueError(
                        "UNSPECIFIED native physical manifest carries authority"
                    )
            elif manifest_state == 1:
                expected_graph_lifecycle = GRAPH_MODES.index(graph_mode) + 1
                if (
                    logical_length != logical_len
                    or physical_length != execution_len
                    or execution_padding_rows != execution_len - logical_len
                    or kv_logical_length < position + logical_len
                    or kv_insert_physical_rows < logical_length
                    or graph_lifecycle != expected_graph_lifecycle
                    or linear_acc_policy not in range(1, 4)
                    or route_count == 0
                ):
                    raise ValueError(
                        "COMPLETE native physical manifest is inconsistent"
                    )
            else:
                raise ValueError("native physical manifest has an unknown state")

            manifest_values = [
                manifest_state, logical_length, physical_length,
                execution_padding_rows, kv_logical_length,
                kv_insert_physical_rows, graph_lifecycle, linear_acc_policy,
                route_count,
            ]
            for (
                family,
                site_id,
                selector,
                flags,
                arguments,
                invocation,
            ) in routes:
                manifest_values.extend((
                    family, site_id, selector, flags, len(arguments), *arguments,
                ))
                if invocation:
                    manifest_values.append(invocation)
            manifest_fingerprint = _physical_manifest_fingerprint(
                manifest_values
            )
            encoded_manifest_fingerprint = (
                manifest_fingerprint_hi << 32
            ) | manifest_fingerprint_lo
            if encoded_manifest_fingerprint != manifest_fingerprint:
                raise ValueError("native physical manifest fingerprint is stale")
            physical_metadata.extend((
                ("capability_attention_ddr_site_count", attention_ddr_site_count),
                ("capability_rope_residency_rank", rope_residency_rank),
                ("descriptor_version", descriptor_version),
                ("graph_lifecycle", graph_lifecycle),
                ("hardware_cost_certificate", 0),
                ("kv_insert_physical_rows", kv_insert_physical_rows),
                ("kv_logical_length", kv_logical_length),
                ("linear_acc_policy", linear_acc_policy),
                ("logical_length", logical_length),
                ("manifest_fingerprint", manifest_fingerprint),
                ("manifest_state", manifest_state),
                ("execution_padding_rows", execution_padding_rows),
                ("physical_length", physical_length),
                ("raw_attention_site_count", raw_attention_site_count),
                ("rope_table_residency", rope_table_residency),
                ("route_count", route_count),
                ("selector_state", 1),  # CAPABILITY_RANKED, not HW-cost certified.
            ))

        fingerprint = (descriptor[5] << 32) | descriptor[6]
        metadata = (
            ("chunk_mode", descriptor[7]),
            ("compute_boundary_policy", descriptor[10]),
            ("compute_num_chunks", counts[2]),
            ("compute_tail_deficit", tail_deficits[2]),
            ("compute_tail_offset", tail_offsets[2]),
            ("input_boundary_policy", descriptor[8]),
            ("input_num_chunks", counts[0]),
            ("input_tail_deficit", tail_deficits[0]),
            ("qkv_boundary_policy", descriptor[9]),
            ("qkv_num_chunks", counts[1]),
            ("qkv_tail_deficit", tail_deficits[1]),
            ("span_count", span_count),
            ("stage_plan_fingerprint", fingerprint),
            *physical_metadata,
        )
        rows.append(StageTuple(*capacities, metadata, descriptor))

    if cursor != len(words):
        raise ValueError("native stage-domain wire has trailing words")
    return tuple(rows)


def _normalise_domain(
    values: Iterable[object],
    *,
    execution_len: int,
) -> tuple[StageTuple, ...]:
    tuples = tuple(StageTuple.from_value(value) for value in values)
    encoded = [item.identity() for item in tuples]
    if encoded != sorted(encoded) or len(set(encoded)) != len(encoded):
        raise PlannerRejectError(
            "CAPABILITY", detail=f"stage tuple domain is unsorted or duplicated at E={execution_len}"
        )
    return tuples


def plan_a6(
    logical_len: int,
    *,
    request: PlannerRequest | Mapping[str, object] | None = None,
    alignment: int = 1,
    physical_limit: int | None = None,
    tuple_domain: Callable[[int], Iterable[object]] | Mapping[int, Iterable[object]] = (),
    feasible: Callable[[A6Candidate], bool | str] | None = None,
    graph_mode: str = GRAPH_RETAINED_CACHE,
    queue_owner_id: int = 0,
    lease_owner_id: int = 0,
    logger: Callable[[Mapping[str, object]], None] | None = None,
    domain_rejections: Mapping[int, str | PlannerRejectError] | None = None,
    max_pairs: int = MAX_A6_PAIRS,
    max_lengths: int = MAX_EXECUTION_LENGTH_CANDIDATES,
    max_diag_records: int = MAX_DIAG_RECORDS,
) -> A6PlanResult:
    """Enumerate and select one deterministic A6 candidate.

    ``feasible`` is the only model-specific hook.  It must return ``True`` to
    accept, or ``False``/a non-empty reason to reject; every other value is a
    programming error.
    """
    logical_len = _checked_int("logical_len", logical_len, minimum=1)
    alignment = _checked_int("alignment", alignment, minimum=1)
    if physical_limit is not None:
        physical_limit = _checked_int("physical_limit", physical_limit, minimum=1)
    max_pairs = _checked_int("max_pairs", max_pairs, minimum=1)
    max_lengths = _checked_int("max_lengths", max_lengths, minimum=1)
    max_diag_records = _checked_int("max_diag_records", max_diag_records, minimum=0)
    queue_owner_id = _checked_int("queue_owner_id", queue_owner_id, minimum=0)
    lease_owner_id = _checked_int("lease_owner_id", lease_owner_id, minimum=0)
    if graph_mode not in GRAPH_MODES:
        raise ValueError(f"graph_mode must be one of {GRAPH_MODES}, got {graph_mode!r}")
    request = request if isinstance(request, PlannerRequest) else PlannerRequest.from_mapping(request)
    lengths, lengths_clamped = _execution_lengths(
        logical_len, alignment=alignment, physical_limit=physical_limit,
        request=request, max_lengths=max_lengths,
    )
    if not lengths:
        raise PlannerRejectError(
            "NO_FEASIBLE", requested=logical_len, limit=physical_limit,
            detail="no execution length fits the physical limit",
        )

    domain_rows: list[tuple[int, tuple[StageTuple, ...]]] = []
    pair_count = 0
    for execution_len in lengths:
        values = tuple_domain(execution_len) if callable(tuple_domain) else tuple_domain.get(execution_len, ())
        remaining = max_pairs - pair_count
        bounded_values = tuple(islice(iter(values), remaining + 1))
        if len(bounded_values) > remaining:
            raise PlannerRejectError(
                "SEARCH_LIMIT", requested=pair_count + len(bounded_values),
                limit=max_pairs, detail="A6 pair domain exceeds limit",
            )
        domain = _normalise_domain(bounded_values, execution_len=execution_len)
        pair_count += len(domain)
        domain_rows.append((execution_len, domain))

    domain_payload = {
        "version": 1,
        "logical_len": logical_len,
        "alignment": alignment,
        "execution_lengths": lengths,
        "tuples": [(e, tuple(item.identity() for item in d)) for e, d in domain_rows],
        "request": request.as_dict(),
        "domain_rejections": tuple(
            sorted((int(key), str(value)) for key, value in (domain_rejections or {}).items())
        ),
    }
    domain_digest = _digest(domain_payload, "RPU-A6-DOMAIN-V1")
    candidates: list[A6Candidate] = []
    selected: A6Candidate | None = None
    feasible_count = 0
    rejected_count = 0
    diag_count = 0
    truncated = False

    def emit(payload: Mapping[str, object]) -> None:
        nonlocal diag_count, truncated
        if logger is None:
            return
        # Bound verbose diagnostics, never the one terminal receipt.
        if payload.get("kind") == "summary":
            logger(payload)
            return
        if diag_count >= max_diag_records:
            truncated = True
            return
        logger(payload)
        diag_count += 1

    for execution_len, reason in sorted((domain_rejections or {}).items()):
        emit({
            "kind": "domain_reject",
            "execution_len": int(execution_len),
            "feasible": False,
            "reject_reason": str(reason),
        })

    for execution_len, domain in domain_rows:
        padding = execution_len - logical_len
        for stage_tuple in domain:
            metadata = dict(stage_tuple.physical_metadata)
            chunks = tuple(
                metadata.get(
                    f"{role}_num_chunks",
                    (execution_len + value - 1) // value,
                )
                for role, value in zip(
                    ("input", "qkv", "compute"), stage_tuple.as_tuple()
                )
            )
            deficits = tuple(
                metadata.get(
                    f"{role}_tail_deficit", n * value - execution_len,
                )
                for role, n, value in zip(
                    ("input", "qkv", "compute"), chunks,
                    stage_tuple.as_tuple(),
                )
            )
            score = (
                max(chunks), padding, max(deficits),
                stage_tuple.input_chunk, stage_tuple.qkv_chunk,
                stage_tuple.compute_chunk,
            )
            if metadata.get("manifest_state") == 1:
                score = (*score,
                         metadata["capability_attention_ddr_site_count"],
                         metadata["capability_rope_residency_rank"],
                         metadata["manifest_fingerprint"])
            if "hardware_cost_score" in metadata:
                score = (metadata["hardware_cost_score"], *score)
            if request.chunk_size is not None and (
                stage_tuple.qkv_chunk != request.chunk_size
                or stage_tuple.compute_chunk != request.chunk_size
            ):
                candidate = A6Candidate(
                    execution_len, padding, stage_tuple, chunks, deficits,
                    score, False,
                    "exact_mismatch",
                )
            else:
                candidate = A6Candidate(execution_len, padding, stage_tuple, chunks, deficits, score, True)
                if feasible is not None:
                    verdict = feasible(candidate)
                    if verdict is True:
                        pass
                    elif verdict is False or (
                        isinstance(verdict, str) and bool(verdict)
                    ):
                        reason = verdict if isinstance(verdict, str) else "infeasible"
                        candidate = A6Candidate(
                            execution_len, padding, stage_tuple, chunks, deficits, score, False, reason,
                        )
                    else:
                        raise TypeError(
                            "feasible callback must return True, False, or a "
                            f"non-empty reject reason, got {verdict!r}"
                        )
            candidates.append(candidate)
            emit({"kind": "candidate", **candidate.as_dict()})
            if candidate.feasible:
                feasible_count += 1
                if selected is None or candidate.score < selected.score:
                    selected = candidate
            else:
                rejected_count += 1

    domain_rejected_count = len(domain_rejections or {})
    total_candidate_count = len(candidates) + domain_rejected_count
    total_rejected_count = rejected_count + domain_rejected_count
    if selected is None:
        emit({
            "kind": "summary", "status": "REJECTED",
            "candidate_count": total_candidate_count,
            "feasible_count": feasible_count,
            "rejected_count": total_rejected_count,
        })
        first_domain_reject = next(
            (
                reason
                for _, reason in sorted((domain_rejections or {}).items())
                if isinstance(reason, PlannerRejectError)
            ),
            None,
        )
        if not candidates and first_domain_reject is not None:
            raise first_domain_reject
        raise PlannerRejectError(
            "EXACT_MISMATCH" if request.mode == REQUEST_EXACT else "NO_FEASIBLE",
            requested=request.chunk_size,
            detail="all A6 candidates were rejected",
        )

    clamped = lengths_clamped
    search_complete = not clamped
    optimality = "EXACT_DOMAIN" if search_complete else "UNKNOWN"
    physical_payload = {
        "graph_mode": graph_mode,
        "queue_owner_id": int(queue_owner_id),
        "lease_owner_id": int(lease_owner_id),
        "execution_len": selected.execution_len,
        "selected_stage_identity": selected.stage_tuple.identity(),
    }
    physical_digest = _digest(physical_payload, "RPU-PHYS-V1")
    plan_digest = _digest(
        {"physical": physical_digest, "request": request.as_dict(), "optimality": optimality},
        "RPU-A6P-V1",
    )
    result = A6PlanResult(
        "ACCEPTED", request, selected, tuple(candidates), total_candidate_count, feasible_count,
        total_rejected_count, search_complete, optimality, graph_mode, int(queue_owner_id),
        int(lease_owner_id), domain_digest, physical_digest, plan_digest, truncated, clamped,
        domain_rejections=tuple(
            sorted((int(key), str(value)) for key, value in (domain_rejections or {}).items())
        ),
    )
    emit({"kind": "summary", **result.as_dict(include_candidates=False)})
    return result


def plan_prefill(
    logical_len: int,
    physical_limit: int,
    padding_budget: int,
    *,
    position: int = 0,
    alignment: int = 1,
    padding_rows: str | int = "auto",
    exact_chunk_size: int | None = None,
    max_stage_chunk_size: int | None = None,
    logger: Callable[[Mapping[str, object]], None] | None = None,
    request_id: object | None = None,
    graph_mode: str = GRAPH_RETAINED_CACHE,
    queue_owner_id: int = 0,
    lease_owner_id: int = 0,
    physical_metadata: Iterable[tuple[str, int]] = (),
    cost_scope: PlannerCostScope | None = None,
    cost_certificates: tuple[PlannerCostCertificate, ...] = (),
    cost_domain_observer=None,
    calibration_selection: PlannerCalibrationSelection | None = None,
    calibration_mint=None,
    resolve_stage_domain: Callable[[int], Iterable[object]],
) -> A6PlanResult:
    """Jointly search execution length and the native three-stage domain.

    Production callers supply ``resolve_stage_domain``.  The callback returns
    the domain-v1 ``int[]`` returned by the RPU ops, including each candidate's
    complete input/QKV/compute schedules, spans, and boundary policies.  Tests
    may also supply structured :class:`StageTuple` rows.  This proves exact
    enumeration only for the three-stage chunk domain:
    kernel/attention routes are still selected by their native capability
    generators, so without an exact-domain hardware cost receipt the aggregate
    result remains ``CAPABILITY_SELECTED``. A sealed receipt ranks only those
    native candidates, not unseen internal routes or end-to-end wall time. The
    full-stage callback is mandatory; there is no scalar compatibility
    selector.
    """
    logical_len = _checked_int("logical_len", logical_len, minimum=1)
    if calibration_selection is not None:
        if not isinstance(calibration_selection, PlannerCalibrationSelection):
            raise ValueError("P7 calibration requires a typed selection")
        if (not calibration_selection.domain_scoped
                and (calibration_selection.scope != cost_scope or cost_certificates)):
            raise ValueError("P7 calibration requires its exact scope and no outer cost certificate")
        if (not calibration_selection.domain_scoped
                and calibration_selection.native_route is not None
                and not callable(calibration_mint)):
            raise ValueError("native calibration requires the actual owner's mint operation")
    physical_limit = _checked_int("physical_limit", physical_limit, minimum=1)
    padding_budget = _checked_int("padding_budget", padding_budget, minimum=0)
    position = _checked_int("position", position, minimum=0)
    alignment = _checked_int("alignment", alignment, minimum=1)
    if padding_rows != "auto":
        if isinstance(padding_rows, bool) or not isinstance(padding_rows, Integral):
            raise ValueError("padding_rows must be 'auto' or a non-negative integer")
        if int(padding_rows) < 0:
            raise ValueError("padding_rows must be 'auto' or a non-negative integer")
        padding_rows = int(padding_rows)
    if exact_chunk_size is not None:
        if (
            isinstance(exact_chunk_size, bool)
            or not isinstance(exact_chunk_size, Integral)
            or int(exact_chunk_size) <= 0
            or int(exact_chunk_size) % 16
        ):
            raise ValueError(
                "exact_chunk_size must be a positive multiple of 16, got "
                f"{exact_chunk_size!r}"
            )
        exact_chunk_size = int(exact_chunk_size)
    if max_stage_chunk_size is not None:
        max_stage_chunk_size = _checked_int(
            "max_stage_chunk_size", max_stage_chunk_size, minimum=1,
        )

    request: dict[str, object] = {
        "mode": REQUEST_EXACT if exact_chunk_size is not None else REQUEST_AUTO,
        "padding_rows": padding_rows,
    }
    if exact_chunk_size is not None:
        request["chunk_size"] = exact_chunk_size
    if padding_rows == "auto":
        request["padding_budget"] = padding_budget

    normalized_request = PlannerRequest.from_mapping(request)
    physical_metadata = tuple(physical_metadata)
    lengths, _ = _execution_lengths(
        logical_len,
        alignment=alignment,
        physical_limit=physical_limit,
        request=normalized_request,
        max_lengths=MAX_EXECUTION_LENGTH_CANDIDATES,
    )

    # Materialise the full search once. A cost-bound winner from an earlier
    # length is re-admitted after selection, without changing the search, so
    # the native latest-oracle proof describes the candidate being dispatched.
    domain: dict[int, tuple[tuple[int, int, int], ...]] = {}
    rejected: dict[int, PlannerRejectError] = {}
    pair_count = 0
    def read_domain(execution_len, remaining):
        raw_values = resolve_stage_domain(int(execution_len))
        if (
            isinstance(raw_values, Sequence)
            and len(raw_values) > 0
            and isinstance(raw_values[0], Integral)
            and not isinstance(raw_values[0], bool)
        ):
            if (
                len(raw_values) >= 2
                and raw_values[0] == 1
                and isinstance(raw_values[1], Integral)
                and not isinstance(raw_values[1], bool)
                and int(raw_values[1]) > remaining
            ):
                raise PlannerRejectError(
                    "SEARCH_LIMIT", requested=MAX_A6_PAIRS - remaining + int(raw_values[1]),
                    limit=MAX_A6_PAIRS,
                    detail="native stage tuple domain exceeds limit",
                )
            raw_rows = raw_values
        else:
            raw_rows = tuple(islice(iter(raw_values), remaining + 1))
            if len(raw_rows) > remaining:
                raise PlannerRejectError(
                    "SEARCH_LIMIT", requested=MAX_A6_PAIRS - remaining + len(raw_rows),
                    limit=MAX_A6_PAIRS, detail="native stage tuple domain exceeds limit")
        if raw_rows and all(
            type(item) is int
            or (isinstance(item, Integral) and not isinstance(item, bool))
            for item in raw_rows
        ):
            rows = _decode_native_stage_domain(
                raw_rows, execution_len=int(execution_len), logical_len=logical_len,
                position=position, graph_mode=graph_mode)
        else:
            rows = raw_rows
        rows = tuple(StageTuple.from_value(row) for row in rows)
        if physical_metadata:
            rows = tuple(StageTuple.from_value(StageTuple(
                *row.as_tuple(), (*row.physical_metadata, *physical_metadata),
                row.physical_descriptor)) for row in rows)
        if len(rows) > remaining:
            raise PlannerRejectError(
                "SEARCH_LIMIT", requested=MAX_A6_PAIRS - remaining + len(rows),
                limit=MAX_A6_PAIRS, detail="native stage tuple domain exceeds limit")
        return rows

    for execution_len in lengths:
        try:
            rows = read_domain(execution_len, MAX_A6_PAIRS - pair_count)
        except PlannerRejectError as exc:
            if exc.code == "SEARCH_LIMIT":
                raise
            # Only a typed candidate rejection belongs to the search domain.
            # Device/programming RuntimeError exceptions must stay visible.
            rejected[int(execution_len)] = exc
            continue
        if not rows:
            rejected[int(execution_len)] = PlannerRejectError(
                "CAPABILITY", requested=int(execution_len),
                detail="native stage-domain oracle returned no feasible tuple",
            )
            continue
        domain[int(execution_len)] = rows  # validated and frozen by plan_a6
        if cost_domain_observer is not None:
            # Query while this exact oracle is current; the next padding length
            # replaces its native admission snapshot. Observation cannot rank.
            cost_domain_observer(int(execution_len), rows)
        pair_count += len(rows)

    def emit(payload: Mapping[str, object]) -> None:
        if logger is None:
            return
        # ``plan_a6`` knows only the stage domain.  Suppress its accepted
        # terminal receipt until the aggregate route-capability label and
        # final digest have been applied below.
        if (
            payload.get("kind") == "summary"
            and payload.get("status") == "ACCEPTED"
        ):
            return
        extra = {"request_id": request_id}
        if payload.get("kind") == "domain_reject":
            extra["position"] = position
        logger({**extra, **payload})

    def feasible(candidate: A6Candidate) -> bool | str:
        if max_stage_chunk_size is not None and max(
            candidate.stage_tuple.qkv_chunk,
            candidate.stage_tuple.compute_chunk,
        ) > max_stage_chunk_size:
            return "stage_chunk_limit"
        # Never launch a final chunk containing only padding rows.
        compute_tail_offset = dict(candidate.stage_tuple.physical_metadata).get(
            "compute_tail_offset",
            (candidate.num_chunks[-1] - 1) * candidate.stage_tuple.compute_chunk,
        )
        if compute_tail_offset >= logical_len:
            return "padding_only_tail"
        return True

    # Derive feasibility with the same planner that will dispatch. Keep rejected
    # rows in diagnostics, but never demand a hardware timing for an illegal
    # padding-only tail or a row outside the actual caller's chunk limit.
    try:
        feasibility = plan_a6(
            logical_len, request=request, alignment=alignment,
            physical_limit=physical_limit, tuple_domain=domain, feasible=feasible,
            domain_rejections=rejected, graph_mode=graph_mode,
            queue_owner_id=queue_owner_id, lease_owner_id=lease_owner_id,
        )
    except PlannerRejectError:
        if logger is not None:
            # Replay diagnostics from the frozen domain, not the native oracle.
            plan_a6(
                logical_len, request=request, alignment=alignment,
                physical_limit=physical_limit, tuple_domain=domain, feasible=feasible,
                domain_rejections=rejected, graph_mode=graph_mode, logger=emit,
                queue_owner_id=queue_owner_id, lease_owner_id=lease_owner_id,
            )
        raise
    feasible_pairs = frozenset(
        (candidate.execution_len, candidate.stage_tuple)
        for candidate in feasibility.candidates if candidate.feasible)
    raw_domain = domain
    if calibration_selection is not None and calibration_selection.domain_scoped:
        target_digest, _ = planner_cost_domain(
            domain, scope=calibration_selection.scope, logical_len=logical_len,
            position=position, graph_mode=graph_mode, feasible_pairs=feasible_pairs)
        if target_digest != calibration_selection.domain_digest:
            # A multi-shape calibration run may visit other real domains. Keep
            # their original owner binding/certificates and normal validation.
            calibration_selection = None
        else:
            if ((cost_scope is not None and cost_scope != calibration_selection.scope)
                    or cost_certificates):
                raise ValueError("P7 calibration requires its exact scope and no outer cost certificate")
            if calibration_selection.native_route is not None and not callable(calibration_mint):
                raise ValueError("native calibration requires the actual owner's mint operation")
            cost_scope = calibration_selection.scope
    domain, cost_certificate, cost_domain_digest, cost_candidate_keys = _apply_cost_certificate(
        domain, scope=cost_scope, logical_len=logical_len,
        certificates=cost_certificates,
        position=position, graph_mode=graph_mode,
        feasible_pairs=feasible_pairs,
    )
    result = plan_a6(
        logical_len,
        request=request,
        alignment=alignment,
        physical_limit=physical_limit,
        tuple_domain=domain,
        feasible=feasible,
        logger=emit,
        domain_rejections=rejected,
        graph_mode=graph_mode,
        queue_owner_id=queue_owner_id,
        lease_owner_id=lease_owner_id,
    )
    result = capability_selected(
        result,
        selection_scope="STAGE_DOMAIN_ROUTE_CAPABILITY_SELECTED",
    )
    selected_cost_key = (
        cost_candidate_keys.get((result.selected.execution_len, result.selected.stage_tuple), "")
        if result.selected is not None else ""
    )
    result = replace(result, cost_scope=cost_scope, cost_domain_digest=cost_domain_digest,
                     selected_cost_candidate_digest=selected_cost_key,
                     cost_feasible_pairs=tuple(
                         (item.execution_len, item.stage_tuple)
                         for item in feasibility.candidates if item.feasible),
                     native_cost_feasible_pairs=tuple(
                         (item.execution_len, item.stage_tuple)
                         for item in feasibility.candidates
                         if item.feasible or (
                             item.reject_reason == "exact_mismatch"
                             and feasible(item) is True)))
    if cost_certificate is not None or result.request.mode == "EXACT":
        # This certificate compares only the fully enumerated native domain.
        # It does not claim unseen inner KV routes or end-to-end wall time are
        # globally optimal, nor override an externally pinned exact chunk.
        external_exact = result.request.mode == "EXACT"
        selection_scope = ("EXTERNALLY_PINNED_NATIVE_CANDIDATE" if external_exact
                           else "MEASURED_NATIVE_DOMAIN_ONLY")
        result = replace(
            result,
            cost_certificate=cost_certificate,
            optimality="EXTERNAL_EXACT" if external_exact else "HARDWARE_COST_CERTIFIED",
            selection_scope=selection_scope,
            plan_digest=_digest({
                "physical": result.physical_plan_digest,
                "request": result.request.as_dict(),
                "cost_receipt": cost_certificate.receipt_sha256 if cost_certificate else None,
                "cost_domain": cost_domain_digest or None,
                "selection_scope": selection_scope,
            }, "RPU-A6P-V1"),
        )
    if calibration_selection is not None:
        if ((result.request.mode != REQUEST_AUTO and not (
                result.request.mode == REQUEST_EXACT and calibration_selection.native_route is not None))
                or not result.search_complete
                or result.clamped or cost_domain_digest != calibration_selection.domain_digest):
            raise ValueError("calibration must retain the complete actual AUTO domain or native EXACT witness")
        # A native route may be timed under its real EXACT chunk request; only
        # rows feasible under that request can execute. It remains private P7
        # calibration below, never an externally authorized/optimality claim.
        matches = [candidate for candidate in result.candidates if candidate.feasible
                   and cost_candidate_keys.get((candidate.execution_len, candidate.stage_tuple))
                   == calibration_selection.candidate_digest]
        if len(matches) != 1:
            raise ValueError("calibration candidate is not feasible in the actual domain")
        candidate = matches[0]
    else:
        candidate = result.selected
    if cost_scope is not None and candidate.execution_len != lengths[-1]:
        current = read_domain(candidate.execution_len, MAX_A6_PAIRS)
        if current != raw_domain[candidate.execution_len]:
            raise ValueError("selected native domain changed during actual re-admission")
    if calibration_selection is not None:
        calibrated_row = candidate.stage_tuple
        if calibration_selection.native_route is not None:
            calibrated_row = calibration_mint(
                candidate.execution_len, calibrated_row, calibration_selection.native_route)
        calibrated_row = StageTuple.from_value(calibrated_row)
        if (calibrated_row.as_tuple() != candidate.stage_tuple.as_tuple()
                or not calibrated_row.physical_descriptor
                or dict(calibrated_row.physical_metadata).get("manifest_state") != 1):
            raise ValueError("calibration requires the same native stage and a COMPLETE descriptor")
        candidate = replace(candidate, stage_tuple=calibrated_row)
        physical_digest = _digest({
            "graph_mode": graph_mode, "queue_owner_id": int(queue_owner_id),
            "lease_owner_id": int(lease_owner_id), "execution_len": candidate.execution_len,
            "selected_stage_identity": candidate.stage_tuple.identity(),
        }, "RPU-PHYS-V1")
        result = replace(
            result, selected=candidate, optimality="P7_CALIBRATION_ONLY",
            selection_scope="P7_CALIBRATION_ONLY", cost_certificate=None,
            selected_cost_candidate_digest=calibration_selection.candidate_digest,
            physical_plan_digest=physical_digest,
            plan_digest=_digest({"physical": physical_digest, "request": result.request.as_dict(),
                                 "selection_scope": "P7_CALIBRATION_ONLY"}, "RPU-A6P-V1"))
    if logger is not None:
        logger({
            "request_id": request_id,
            "kind": "summary",
            **result.as_dict(include_candidates=False),
        })
    return result


def capability_selected(
    result: A6PlanResult, *, selection_scope: str
) -> A6PlanResult:
    """Label a native-selected finite-domain bridge without claiming A6 optimality."""
    if not isinstance(selection_scope, str) or not selection_scope:
        raise ValueError("selection_scope must be a non-empty string")
    optimality = "CAPABILITY_SELECTED"
    return replace(
        result,
        optimality=optimality,
        selection_scope=selection_scope,
        plan_digest=_digest(
            {
                "physical": result.physical_plan_digest,
                "request": result.request.as_dict(),
                "optimality": optimality,
                "selection_scope": selection_scope,
            },
            "RPU-A6P-V1",
        ),
    )


def plan_fixed_component_execution(
    logical_len: int,
    *,
    execution_len: int,
    chunk_size: int,
    component_id: str,
    stage: str,
    generation: int = 0,
    position: int = 0,
    stage_config: Mapping[str, object] | None = None,
    graph_mode: str = GRAPH_COMPOSITE_CHILD,
    physical_metadata: Iterable[tuple[str, int]] = (),
    physical_descriptor: Iterable[int] = (),
    queue_owner_id: int = 0,
    lease_owner_id: int = 0,
) -> A6PlanResult:
    """Describe and validate one fixed native component geometry.

    This is deliberately a one-row A6 domain: it gives fixed-ABI children the
    same immutable receipt and graph-key codec as searched children without
    claiming that a search took place.  An exact caller chunk must equal the
    native singleton; ``auto`` simply selects that singleton.
    """
    logical_len = _checked_int("logical_len", logical_len, minimum=1)
    execution_len = _checked_int("execution_len", execution_len, minimum=1)
    chunk_size = _checked_int("chunk_size", chunk_size, minimum=1)
    generation = _checked_int("generation", generation, minimum=0)
    position = _checked_int("position", position, minimum=0)
    if execution_len < logical_len:
        raise ValueError("execution_len must be >= logical_len")
    if chunk_size % 16:
        raise ValueError("fixed component chunk_size must be a multiple of 16")
    if not isinstance(component_id, str) or not component_id:
        raise ValueError("component_id must be a non-empty string")
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a non-empty string")

    raw_config = {} if stage_config is None else dict(stage_config)
    unknown = set(raw_config) - {"chunk_size"}
    if unknown:
        raise ValueError(
            "fixed component stage config has unsupported field(s): "
            f"{sorted(map(str, unknown))}"
        )
    requested_chunk = raw_config.get("chunk_size", "auto")
    request: dict[str, object] = {
        "mode": REQUEST_AUTO if requested_chunk == "auto" else REQUEST_EXACT,
        "padding_rows": execution_len - logical_len,
    }
    if requested_chunk != "auto":
        request["chunk_size"] = requested_chunk

    result = plan_a6(
        logical_len,
        request=request,
        alignment=1,
        physical_limit=execution_len,
        tuple_domain={
            execution_len: (
                StageTuple(
                    chunk_size,
                    chunk_size,
                    chunk_size,
                    physical_metadata=(
                        (f"component:{component_id}", 1),
                        ("execution_generation", generation),
                        ("fixed_abi_singleton", 1),
                        ("position", position),
                        (f"stage:{stage}", 1),
                        *tuple(physical_metadata),
                    ),
                    physical_descriptor=tuple(physical_descriptor),
                ),
            ),
        },
        graph_mode=graph_mode,
        queue_owner_id=queue_owner_id,
        lease_owner_id=lease_owner_id,
    )
    return capability_selected(
        result, selection_scope="FIXED_ABI_SINGLETON_NO_SEARCH",
    )


def publish_component_execution_receipt(
    owner,
    plan: A6PlanResult,
    *,
    component: str,
    stage: str,
    generation: int,
    logical_len: int,
    resolved_chunk_size: int,
    extra_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Publish the common immutable-plan receipt fields for one component."""
    receipt = plan.as_dict(include_candidates=False)
    descriptor = (
        () if plan.selected is None
        else plan.selected.stage_tuple.physical_descriptor
    )
    receipt.update({
        "component": component,
        "stage": stage,
        "generation": int(generation),
        "logical_len": int(logical_len),
        "execution_len": int(plan.selected.execution_len),
        "chunk_size": int(resolved_chunk_size),
        "descriptor_authority": bool(descriptor),
    })
    if extra_fields is not None:
        receipt.update(extra_fields)
    vars(owner)["_rpu_last_execution_plan"] = receipt
    return receipt


__all__ = [
    "A6Candidate", "A6PlanResult", "GRAPH_BOUNDED_ONESHOT",
    "GRAPH_COMPOSITE_CHILD", "GRAPH_MODES", "GRAPH_NATIVE_COMPOSITE1",
    "GRAPH_RETAINED_CACHE", "MAX_A6_PAIRS", "MAX_DIAG_RECORDS",
    "MAX_EXECUTION_LENGTH_CANDIDATES", "PlannerRejectError", "PlannerRequest",
    "PlannerCostScope", "PlannerCostCertificate", "planner_cost_domain",
    "StageTuple", "capability_selected", "native_chunk_reject", "plan_a6",
    "plan_fixed_component_execution", "plan_prefill",
    "publish_component_execution_receipt",
]
