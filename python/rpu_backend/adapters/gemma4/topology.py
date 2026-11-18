"""Gemma4-E4B cold core topology (pure Python, import-safe).

``rpu_execution.model.num_cores`` is a compute budget of 4, 6 or 8 (default 8).
It selects one fixed component layout; there is no per-component override.

======  =========  ===  ===  ====  ======================================
budget  attention  MLP  PLE  head  notes
======  =========  ===  ===  ====  ======================================
8       2          8    8    8     original eight-core layout, unchanged
6       2          6    4    6     MLP zero-padded 10240 -> 10272; head 262144 -> 262176
4       2          4    4    4
======  =========  ===  ===  ====  ======================================

The six-lane padding follows the projection storage: a row-partitioned Down
needs K/6 to hold whole swizzle granules of 16 FP16, 32 INT8 or 64 packed-W4
channels, so the on-install W8A16 and W4A16 recipes both use 10368.

Attention keeps two lanes (two KV heads) at every budget. The per-layer
embedding (PLE, width 256 = 16 FP16 granules) splits evenly over four or eight
lanes but not six: at budget 6 it keeps the budget-4 shapes and its four
partial producers feed the six-lane ring through the existing preparation.
Physical KV/DDR striping stays eight lanes, so the KV cache is unchanged.
The tied lm_head (vocab 262144) is an ordinary RPU Linear outside the native
owner and uses the full compute budget (8, 6 or 4 lanes; at 6 lanes, rows are
zero-padded 262144 -> 262176 before swizzle and sliced back to logical vocab).
The native owner mirrors this table in ``src/core/gemma4_execution_topology.h``.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

PHYSICAL_KV_CORES = 8
ATTENTION_CORES = 2
TOPOLOGY_SCHEMA = 1
CORE_BUDGETS = (4, 6, 8)
E4B_INTERMEDIATE = 10240
E4B_PLE_DIM = 256
# FP16 column/row partitions need 16 channels per lane.
_FP16_LANE_GRANULE = 16
# Per-lane K granule of each projection storage (None = FP16): the INT8 swizzle
# moves 32-byte rows of 32 channels; packed W4 swizzles 16N x 64K tiles.
MLP_LANE_GRANULE = {None: _FP16_LANE_GRANULE, "w8a16": 32, "w4a16": 64}


def ple_cores(num_cores: int) -> int:
    """PLE lanes for a budget: eight at 8, four at 4 and 6."""
    return 8 if require_core_budget(num_cores) == 8 else 4


def require_core_budget(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in CORE_BUDGETS:
        raise ValueError("Gemma4 model.num_cores must be one of 4, 6 or 8")
    return int(value)


def mlp_physical_intermediate(logical: int, mlp_tp: int, quantization=None) -> int:
    """Physical MLP width: six lanes round up to 6 storage granules; others keep logical."""
    if mlp_tp != 6:
        return int(logical)
    alignment = 6 * MLP_LANE_GRANULE[quantization]
    return (int(logical) + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Gemma4Topology:
    num_cores: int
    attn_tp: int
    mlp_tp: int
    ple_tp: int
    logical_intermediate: int
    physical_intermediate: int
    physical_kv_cores: int = PHYSICAL_KV_CORES

    def native_readback(self) -> tuple[int, ...]:
        """Expected ``torch.ops.rpu.gemma4_get_execution_topology`` result."""
        return (TOPOLOGY_SCHEMA, self.num_cores, self.attn_tp, self.mlp_tp,
                self.ple_tp, self.physical_kv_cores, self.logical_intermediate,
                self.physical_intermediate)

    @property
    def mlp_padding(self) -> int:
        return self.physical_intermediate - self.logical_intermediate

    @property
    def head_tp(self) -> int:
        """Tied lm_head lanes; not part of the native owner's readback."""
        return self.num_cores


def resolve_gemma4_topology(num_cores=8, *, intermediate_size: int = E4B_INTERMEDIATE,
                            ple_dim: int = E4B_PLE_DIM,
                            num_kv_heads: int = 2, quantization=None) -> Gemma4Topology:
    """Fixed layout for an admitted E4B geometry; anything else fails closed.

    ``quantization`` (None, ``"w8a16"`` or ``"w4a16"``) selects the projection
    storage, which sets the six-lane physical MLP width.
    """
    budget = require_core_budget(num_cores)
    if quantization not in MLP_LANE_GRANULE:
        raise ValueError("Gemma4 quantization must be None, 'w8a16' or 'w4a16'")
    if (budget != 8 or quantization is not None) and (
            intermediate_size != E4B_INTERMEDIATE or ple_dim != E4B_PLE_DIM
            or num_kv_heads != ATTENTION_CORES):
        raise ValueError("Gemma4 reduced core budgets and quantization admit only the "
                         "E4B text geometry")
    mlp_tp = budget
    ple_tp = ple_cores(budget)
    physical = mlp_physical_intermediate(intermediate_size, mlp_tp, quantization)
    if (physical % (_FP16_LANE_GRANULE * mlp_tp)
            or ple_dim % (_FP16_LANE_GRANULE * ple_tp)):
        raise ValueError("Gemma4 geometry does not fit its fixed core layout")
    return Gemma4Topology(budget, ATTENTION_CORES, mlp_tp, ple_tp,
                          int(intermediate_size), physical)


def execution_num_cores(execution) -> int:
    """Budget from a normalized ``rpu_execution`` mapping (omitted -> 8)."""
    model = (execution or {}).get("model") or {}
    return require_core_budget(model.get("num_cores", 8))


def require_same_core_budget(old_execution, new_execution) -> None:
    if execution_num_cores(old_execution) != execution_num_cores(new_execution):
        raise ValueError("model.num_cores is cold-only: reload a fresh model to change topology")


def pad_mlp_projection(weight, *, role: str, physical: int):
    """Zero-pad one CPU projection to the physical width before its swizzle.

    gate/up gain zero output rows; down gains zero input columns. The zero
    gate rows make GELU-tanh(0) * up = 0, and the zero down columns ignore those
    channels, so the padded MLP computes exactly the logical result. INT8 codes
    pad the same way; their scales are padded by the caller.
    """
    import torch

    if role not in ("gate", "up", "down"):
        raise ValueError("Gemma4 MLP padding requires gate/up/down")
    axis = 1 if role == "down" else 0
    if weight.ndim != 2 or weight.shape[axis] > physical:
        raise ValueError("Gemma4 MLP padding requires a 2D projection no wider than physical")
    if weight.shape[axis] == physical:
        return weight
    shape = list(weight.shape)
    shape[axis] = physical
    padded = weight.new_zeros(shape)
    padded[:weight.shape[0], :weight.shape[1]].copy_(weight)
    return padded
