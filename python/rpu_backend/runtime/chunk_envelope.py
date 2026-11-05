"""Shared chunk-envelope value and lookup; adapters own safe chunk ceilings.

The envelope bounds the native planner's search so it cannot select chunks
outside the model's SPM capacity. Missing geometries are rejected before prefill.
Keeping model-specific rows in their adapters avoids architecture dependencies
in this shared runtime module.
"""
from __future__ import annotations


class ChunkEnvelope(tuple):
    """(max_kv_len, chunk). Named for readability at the declaration sites.

    chunk       ceiling on the chosen chunk. Safety is a ceiling property: SPM
                footprint is monotone in chunk size, so at-or-below a
                measured-safe chunk is safe a fortiori. 0 = "auto is certified
                within max_kv_len".
    max_kv_len  installed storage/position capacity, bound before native planning.
                Adapter tables use 0 for an unbound capacity; their chunk ceiling
                remains independent of context length. Per-request native checks
                still account for length-dependent masks, SPM and kernel limits.
    """
    __slots__ = ()

    def __new__(cls, max_kv_len: int, chunk: int = 0):
        return super().__new__(cls, (int(max_kv_len), int(chunk)))

    max_kv_len = property(lambda self: self[0])
    chunk = property(lambda self: self[1])

    def bind_capacity(self, capacity: int) -> "ChunkEnvelope":
        """Bind actual cold-installed capacity without a historical length cap."""
        from numbers import Integral
        if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity <= 0:
            raise ValueError("chunk envelope capacity must be a positive integer")
        return ChunkEnvelope(capacity, self.chunk)


def make_lookup(table, table_location: str):
    """Build the `chunk_envelope_for` callable that decoder.py expects.

    `table` maps (arch, num_layers, hidden_size) -> ChunkEnvelope. Raising on a
    miss is the feature, not an inconvenience: an unmeasured geometry must not
    reach the SPM planner. `table_location` names the file to edit, so the error
    can say where to add the row.
    """
    def chunk_envelope_for(arch: str, num_layers: int,
                           hidden_size: int) -> ChunkEnvelope:
        key = (arch, int(num_layers), int(hidden_size))
        env = table.get(key)
        if env is None:
            raise RuntimeError(
                f"no certified chunk envelope for {key} "
                f"(arch, num_layers, hidden_size). Validate the SPM capacity "
                f"and add the supported bound to {table_location} before running "
                f"this profile. See docs/runtime_config.md."
            )
        return env
    return chunk_envelope_for
