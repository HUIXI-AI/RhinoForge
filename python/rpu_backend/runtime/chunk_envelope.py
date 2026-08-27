"""Certified chunk-envelope value and lookup helpers.

Each adapter owns its exact profile rows, while this module remains independent
of model architectures. Missing rows are rejected before planning because an
unverified chunk may exceed the model's SPM envelope. See
``docs/model_porting.md#8-verification-gates``.
"""
from __future__ import annotations


class ChunkEnvelope(tuple):
    """(max_kv_len, chunk). Named for readability at the declaration sites.

    chunk       ceiling on the chosen chunk. Safety is a ceiling property: SPM
                footprint is monotone in chunk size, so at-or-below a
                measured-safe chunk is safe a fortiori. 0 = "auto is certified
                within max_kv_len".
    max_kv_len  bounds the automatic search space, which grows with the rounded
                sequence length. On the explicit-mask path it also bounds the
                length-scaled attention-mask buffer. A pinned row can therefore
                admit a larger length than an automatic row.
    """
    __slots__ = ()

    def __new__(cls, max_kv_len: int, chunk: int = 0):
        return super().__new__(cls, (int(max_kv_len), int(chunk)))

    max_kv_len = property(lambda self: self[0])
    chunk = property(lambda self: self[1])


def make_lookup(table, table_location: str):
    """Build the `chunk_envelope_for` callable that decoder.py expects.

    `table` maps (arch, num_layers, hidden_size) -> ChunkEnvelope. Raising on a
        miss is deliberate: an unverified geometry must not reach the SPM planner.
        ``table_location`` names the adapter table to update after verification.
    """
    def chunk_envelope_for(arch: str, num_layers: int,
                           hidden_size: int) -> ChunkEnvelope:
        key = (arch, int(num_layers), int(hidden_size))
        env = table.get(key)
        if env is None:
            raise RuntimeError(
                f"no certified chunk envelope for {key} "
                f"(arch, num_layers, hidden_size). Verify the exact profile on "
                f"hardware, then add its row to {table_location}. See "
                "docs/model_porting.md#8-verification-gates."
            )
        return env
    return chunk_envelope_for
