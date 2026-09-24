"""Structured PaliGemma2 fallback events."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rpu_backend.api.errors import RPUBackendError

ALLOWED_EVENTS = frozenset({
    ("paligemma_outer", "attention_mask_mapping", "prefill"),
    ("paligemma_outer", "attention_mask_mapping", "decode"),
    ("paligemma_outer", "placeholder_mask_creation", "prefill"),
    ("paligemma_outer", "masked_scatter_merge", "prefill"),
    ("gemma2", "cpu_embed_tokens", "decode"),
    ("lm_head", "cpu_lm_head", "prefill"),
    ("lm_head", "cpu_lm_head", "decode"),
    ("shape_index", "position_or_cache_index", "prefill"),
    ("shape_index", "position_or_cache_index", "decode"),
})


def _validate_counter(name: str, value: int, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RPUBackendError(f"PaliGemma2 fallback {name} must be an int")
    if value < minimum:
        raise RPUBackendError(
            f"PaliGemma2 fallback {name} must be >= {minimum}"
        )
    return value


def _normalize_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_shape(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_shape(item) for item in value]
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if hasattr(value, "shape"):
        try:
            return [_normalize_shape(item) for item in list(value.shape)]
        except TypeError:
            return repr(value)
    return repr(value)


@dataclass(frozen=True)
class FallbackEvent:
    component: str
    op: str
    stage: str
    allowed: bool
    reason: str
    shape: Any
    dtype: str
    device_from: str
    device_to: str
    bytes_moved: int
    call_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FallbackRecorder:
    def __init__(self) -> None:
        self.events: list[FallbackEvent] = []

    def record(
        self,
        *,
        component: str,
        op: str,
        stage: str,
        reason: str,
        shape: Any,
        dtype: str,
        device_from: str,
        device_to: str,
        bytes_moved: int,
        call_count: int = 1,
    ) -> None:
        bytes_moved = _validate_counter(
            "bytes_moved",
            bytes_moved,
            minimum=0,
        )
        call_count = _validate_counter("call_count", call_count, minimum=1)
        allowed = (component, op, stage) in ALLOWED_EVENTS
        self.events.append(FallbackEvent(
            component=component,
            op=op,
            stage=stage,
            allowed=allowed,
            reason=reason,
            shape=_normalize_shape(shape),
            dtype=dtype,
            device_from=device_from,
            device_to=device_to,
            bytes_moved=bytes_moved,
            call_count=call_count,
        ))

    def disallowed_count(self) -> int:
        return sum(1 for event in self.events if not event.allowed)

    def assert_no_disallowed(self) -> None:
        bad = [event for event in self.events if not event.allowed]
        if bad:
            first = bad[0]
            raise RPUBackendError(
                "PaliGemma2 disallowed fallback event: "
                f"{first.component}.{first.op} at stage={first.stage} "
                f"reason={first.reason!r}"
            )

    def aggregate(self) -> dict[tuple[str, str, str, str], dict[str, int]]:
        out: dict[tuple[str, str, str, str], dict[str, int]] = {}
        for event in self.events:
            key = (event.component, event.op, event.stage, event.reason)
            row = out.setdefault(key, {"call_count": 0, "bytes_moved": 0})
            row["call_count"] += event.call_count
            row["bytes_moved"] += event.bytes_moved
        return out
