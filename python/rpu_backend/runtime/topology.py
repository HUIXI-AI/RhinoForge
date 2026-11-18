"""Cold model topology shared by planning, weight layout and KV allocation.

The execution budget counts participating compute cores. DDR/KV controller
striping is a physical ABI and remains eight even with a smaller budget.
Architecture adapters own profile admission; this module owns geometry.
"""
from dataclasses import dataclass
from numbers import Integral


PHYSICAL_CORE_COUNT = 8
DECODER_TEMPLATE_VERSION = 2
# These are bounded capability templates, not performance winners.
_DECODER_TEMPLATES = {
    4: ((4, 4, 4),),
    6: ((4, 6, 4), (4, 6, 6), (4, 4, 4)),
    8: ((8, 8, 8),),
}


@dataclass(frozen=True)
class DecoderGeometryProfile:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    six_core_mlp_tp: int
    # Per-head Q/K RMSNorm before RoPE. Adapters admit only rows whose decoder
    # structure matches their architecture.
    qk_norm: bool = True
    six_core_lm_head_tp: int = 4

    def geometry(self):
        return {name: getattr(self, name) for name in (
            "hidden_size", "intermediate_size", "num_q_heads", "num_kv_heads",
            "head_dim", "vocab_size")}

    def weight_geometry(self):
        return {"num_layers": self.num_layers, **self.geometry()}


# Geometry only: architecture/precision/asset admission stays with the adapter.
# Six-core 4B uses a padded physical intermediate; logical model admission
# remains exact. Padding is a one-time weight conversion, not a forward op.
DECODER_GEOMETRY_PROFILES = (
    DecoderGeometryProfile(1024, 3072, 28, 16, 8, 128, 151936, 6, six_core_lm_head_tp=6),
    DecoderGeometryProfile(2048, 6144, 28, 16, 8, 128, 151936, 6, six_core_lm_head_tp=6),
    DecoderGeometryProfile(2560, 9728, 36, 32, 8, 128, 151936, 6, six_core_lm_head_tp=6),
    DecoderGeometryProfile(4096, 12288, 36, 32, 8, 128, 151936, 6, six_core_lm_head_tp=6),
    # Large dense Qwen3 models use the same reduced execution template:
    # attention remains on four lanes while MLP and head use six-lane padded widths.
    DecoderGeometryProfile(5120, 17408, 40, 40, 8, 128, 151936, 6, six_core_lm_head_tp=6),
    DecoderGeometryProfile(5120, 25600, 64, 64, 8, 128, 151936, 6, six_core_lm_head_tp=6),
    # Llama-3.2-1B reuses the same template: attention 4 (8 Q + 2 KV heads per
    # core, D64, no QK-norm), MLP6 on a padded 8256 physical width, and head 6
    # (128256 = 6 x 21376).
    DecoderGeometryProfile(2048, 8192, 16, 32, 8, 64, 128256, 6, qk_norm=False, six_core_lm_head_tp=6),
)


def decoder_mlp_intermediate_size(logical_size: int, mlp_tp: int) -> int:
    """Physical width for an already admitted cold decoder topology."""
    return {3584: 3648, 8192: 8256, 9728: 9792, 17408: 17472,
            25600: 25728, 18944: 19008, 11008: 11136}.get(logical_size, logical_size) if mlp_tp == 6 else logical_size


def decoder_lm_head_vocab_size(logical_size: int, lm_head_tp: int) -> int:
    """Physical vocabulary rows for an already admitted cold decoder topology."""
    return 151968 if lm_head_tp == 6 and logical_size == 151936 else logical_size


# Signed W4 pgrp packing uses the unified N32K32 ABI. A row-partitioned Down
# therefore needs K/TP % 32. Six lanes need only a 192-aligned physical width;
# 4/8 lanes and already aligned widths keep logical storage.
PACKED_W4_K_TILE = 32
NVFP4_K_TILE = 64


def decoder_packed_w4_mlp_intermediate_size(logical_size: int, mlp_tp: int) -> int:
    """Physical MLP width for a packed signed-W4 G32 decoder (native mirror)."""
    if mlp_tp != 6:
        return logical_size
    alignment = PACKED_W4_K_TILE * mlp_tp
    return (logical_size + alignment - 1) // alignment * alignment


def decoder_nvfp4_mlp_intermediate_size(logical_size: int, mlp_tp: int) -> int:
    """Physical MLP width for a dense NVFP4 decoder (64-wide per-lane K tile)."""
    if mlp_tp != 6:
        return logical_size
    alignment = NVFP4_K_TILE * mlp_tp
    return (logical_size + alignment - 1) // alignment * alignment


def decoder_geometry_profile(*, hidden_size, intermediate_size, num_q_heads,
                             num_kv_heads, head_dim, vocab_size):
    geometry = dict(hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        vocab_size=vocab_size)
    if any(isinstance(value, bool) or not isinstance(value, Integral)
           for value in geometry.values()):
        return None
    return next((profile for profile in DECODER_GEOMETRY_PROFILES
                 if profile.geometry() == geometry), None)


def execution_core_count(execution) -> int:
    value = (execution or {}).get("model", {}).get("num_cores", PHYSICAL_CORE_COUNT)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("rpu_execution model.num_cores must be an integer")
    if value not in _DECODER_TEMPLATES:
        raise ValueError("rpu_execution model.num_cores must be one of 4, 6 or 8")
    return int(value)


@dataclass(frozen=True)
class DecoderTopology:
    num_cores: int
    attn_tp: int
    mlp_tp: int
    lm_head_tp: int
    physical_kv_cores: int = PHYSICAL_CORE_COUNT

    def __post_init__(self):
        budget = execution_core_count({"model": {"num_cores": self.num_cores}})
        values = (self.attn_tp, self.mlp_tp, self.lm_head_tp, self.physical_kv_cores)
        if (any(isinstance(v, bool) or not isinstance(v, Integral) for v in values)
                or values[:3] not in _DECODER_TEMPLATES[budget]
                or self.physical_kv_cores != PHYSICAL_CORE_COUNT):
            raise ValueError("decoder topology must match its fixed core template")

    def identity(self) -> tuple[int, ...]:
        return (DECODER_TEMPLATE_VERSION, self.num_cores, self.attn_tp, self.mlp_tp,
                self.lm_head_tp, self.physical_kv_cores)


def resolve_decoder_topology(*, num_cores: int, hidden_size: int,
                             intermediate_size: int, num_q_heads: int,
                             num_kv_heads: int, head_dim: int,
                             vocab_size: int) -> DecoderTopology:
    """Validate the fixed capability template without padding model weights."""
    budget = execution_core_count({"model": {"num_cores": num_cores}})
    geometry = (hidden_size, intermediate_size, num_q_heads,
                num_kv_heads, head_dim, vocab_size)
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0
           for v in geometry):
        raise ValueError("decoder topology dimensions must be positive integers")
    if hidden_size % 16 or intermediate_size % 16 or head_dim % 16 or vocab_size % 16:
        raise ValueError("decoder topology requires FP16 dimensions aligned to 16")
    if num_q_heads % num_kv_heads:
        raise ValueError("decoder topology requires integral grouped-query attention")

    profile = decoder_geometry_profile(hidden_size=hidden_size,
        intermediate_size=intermediate_size, num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads, head_dim=head_dim, vocab_size=vocab_size)
    if budget != PHYSICAL_CORE_COUNT and profile is None:
        raise ValueError("no admitted reduced-core decoder template for this geometry")
    attn, mlp, lm_head = _DECODER_TEMPLATES[budget][0]
    if budget == 6:
        mlp = profile.six_core_mlp_tp
        lm_head = profile.six_core_lm_head_tp
    physical_intermediate = decoder_mlp_intermediate_size(intermediate_size, mlp)
    physical_vocab = decoder_lm_head_vocab_size(vocab_size, lm_head)
    if (num_q_heads % attn or num_kv_heads % attn
            or physical_intermediate % (16 * mlp) or physical_vocab % (16 * lm_head)):
        raise ValueError("decoder geometry does not fit the fixed core template")
    return DecoderTopology(budget, attn, mlp, lm_head)



def require_same_decoder_topology(old_execution, new_execution) -> None:
    if execution_core_count(old_execution) != execution_core_count(new_execution):
        raise ValueError("model.num_cores is cold-only: reload a fresh model to change topology")


def decoder_topology_for_model(model):
    """Read the installed decoder authority and verify an optional outer view."""
    outer = getattr(model, "_rpu_decoder_topology", None)
    inner = getattr(model, "model", None)
    installed = getattr(inner, "_rpu_decoder_topology", None)
    if outer is not None and installed is not None and outer != installed:
        raise ValueError("outer model and installed decoder topology differ")
    if (outer is not None and inner is not None
            and getattr(inner, "_rpu_decoder_handle", None) is not None and installed is None):
        raise ValueError("installed decoder is missing its cold model topology")
    return installed if installed is not None else outer


def validate_decoder_cache_topology(topology, cache, *, batch_size=None) -> None:
    if topology is None:
        return
    if topology.num_cores != PHYSICAL_CORE_COUNT and (
            getattr(cache, "batch_size", 1) != 1 or batch_size not in (None, 1)):
        raise ValueError("reduced-core decoder pilot requires batch_size=1")
    if (getattr(cache, "attn_tp", None) != topology.attn_tp or
            getattr(cache, "physical_kv_cores", None) != topology.physical_kv_cores):
        raise ValueError("RPUCache topology does not match the installed decoder; "
                         "construct it with RPUCache.from_model(model, ...)")
