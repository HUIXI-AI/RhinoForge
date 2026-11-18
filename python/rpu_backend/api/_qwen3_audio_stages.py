"""Exclusive per-stage wall time for the speech request pipelines.

A total request latency cannot say whether the cost is the autoregression, the
audio encoder or the codec decode, so each stage is timed where it is installed.
Both speech APIs report the result through ``execution_stats()["stage_ms"]``.
"""
from __future__ import annotations

from contextlib import contextmanager
import time


class StageTimer:
    """Exclusive wall time per stage: nested time is charged to the inner stage only.

    TTS runs the Predictor's sampling inside the Talker's loop, so plain elapsed
    time per stage would count the same nanoseconds twice. Charging a nested
    interval to the innermost open stage keeps the reported values disjoint,
    which makes them both comparable to each other and additive against the
    request's own wall time. Stages that never nest are unaffected.

    ``names`` fixes the reported stage set and its pipeline order; ``label``
    only names the owner in error messages.
    """

    def __init__(self, label: str, names):
        self._label = label
        self._names = tuple(names)
        self._totals, self._open = {}, []

    def reset(self):
        self._totals, self._open = {}, []

    @contextmanager
    def stage(self, name):
        now = time.perf_counter_ns()
        if self._open:
            parent, since = self._open[-1]
            self._totals[parent] = self._totals.get(parent, 0) + now - since
        self._open.append((name, now))
        try:
            yield
        finally:
            now = time.perf_counter_ns()
            name, since = self._open.pop()
            self._totals[name] = self._totals.get(name, 0) + now - since
            if self._open:
                self._open[-1] = (self._open[-1][0], now)

    def milliseconds(self, total_ns):
        """Every known stage plus the unattributed remainder, in pipeline order.

        A stage that did not run reports 0.0 instead of disappearing, so a cached
        speaker embedding stays visible rather than changing the reported stage set.
        """
        if self._open:
            raise RuntimeError(f"{self._label} stage timer has an unclosed stage")
        unknown = sorted(set(self._totals) - set(self._names))
        if unknown:
            raise RuntimeError(f"{self._label} stage timer recorded unknown stages: {unknown}")
        out = {name: self._totals.get(name, 0) / 1e6 for name in self._names}
        # The glue outside every stage: text split, reference read, embedding
        # lookups, composition and the CPU copies between stages. Never
        # negative, because every stage lies inside the request.
        out["other"] = (total_ns - sum(self._totals.values())) / 1e6
        return out
