"""Shared chunk-envelope value and lookup; each adapter owns its admitted rows.

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
    max_kv_len  bounds the AUTO SEARCH SPACE, because compute_chunks_impl:513
                searches up to hi = ceil16(seq_len) — a longer prefill lets auto
                reach a bigger, never-measured chunk. (On the explicit-mask path
                it additionally bounds sdpa_mask, the only length-scaled buffer
                in declare_buffers.) This is why a PINNED row can afford a
                generous length and an AUTO row cannot.
    """
    __slots__ = ()

    def __new__(cls, max_kv_len: int, chunk: int = 0):
        return super().__new__(cls, (int(max_kv_len), int(chunk)))

    max_kv_len = property(lambda self: self[0])
    chunk = property(lambda self: self[1])


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
                f"(arch, num_layers, hidden_size). This combination has never "
                f"been measured on hardware, and running it on the auto chunk "
                f"planner risks busting SPM and WEDGING the board. Measure it on "
                f"a RESETTABLE board, then add the row to {table_location} AND to "
                f"docs/roadmap/chunk_certified_envelope.md — the two change "
                f"together."
            )
        return env
    return chunk_envelope_for
