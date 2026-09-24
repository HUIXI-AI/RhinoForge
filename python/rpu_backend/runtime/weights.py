"""rpu_backend.runtime.weights — canonical weight primitives.

Per ADR §6.4 + DELETION-LEDGER §D3 (v5.0 final form). Consolidates
`_internal/weights.py` + `core/weights/{swizzle,partition,fp16_cast,__init__}.py`
into a single flat module. Module-level symbols only (no nested package).

Per D-04: exports `swizzle_linear`, `swizzle_model_inplace`, `LinearPartition`,
`get_linear_partition`. Compat aliases `transform_linear_weight` +
`convert_linear_weights_inplace` retained for sibling converter back-compat.

Pitfall P1 discipline: `SKIP_LINEAR_NAMES = {'dense'}` is the AdaRMS
double-swizzle guard — do not modify without reading
docs/pitfalls.md#p1--double-swizzle.
"""
from __future__ import annotations

import copy
import types
from collections.abc import Mapping
from functools import partial
from numbers import Integral
from typing import Dict, Any

import torch
import torch.nn as nn

from rpu_backend.runtime.log import _LOG
from rpu_backend.runtime.topology import PHYSICAL_CORE_COUNT


NUM_CORES = PHYSICAL_CORE_COUNT

# Projection child names pinned to ROW partition regardless of shape, because
# that is what the fused SPM decoders launch them with (see
# `src/fused/rpu_dinov3_vision_model.cpp:66` — "q/k/v/up — col-partition,
# o/down — row-partition", and every `rpu_launch_linear_spm_to_spm_*` site for
# o_proj / down_proj). Single source of truth: `adapters/pi05/w4pack.py`
# derives its int4 packing table from this rather than keeping a second copy.
ROW_PARTITION_NAMES = frozenset({"o_proj", "down_proj"})

# The layout recorded for a Linear this converter deliberately did NOT swizzle
# (`SKIP_LINEAR_NAMES`). It is a layout, not the absence of one: the weight is
# still ROW-MAJOR, and eager `aten::linear` would read those bytes as swizzled.
# Written to `partition_info` and to `linear._rpu_linear_partition`, and read by
# `_guarded_linear_forward` at the moment of the hazard.
SKIPPED_PARTITION = -2

# The layout recorded for a Linear whose shape `get_linear_partition` refused
# (return -1, "no legal row or col split at this core count / element width").
# The converter leaves that weight ROW-MAJOR too, so it is the same class of
# decision as SKIPPED_PARTITION and needs the same record — but a DIFFERENT
# sentinel, because the reason differs and so does the diagnosis.
#
# It is not automatically a hazard: when eager `aten::linear` re-derives -1 as
# well, `rpu_linear` warns once and falls back to CPU, which is the correct
# answer for a row-major weight. The hazard is the DISAGREEMENT, and it is
# reachable two ways: (a) `force_col_partition=True` demotes a legal row
# partition to -1 while the eager shape rule still says row, and (b) the
# converter derives with the WEIGHT's element width (1 byte for W8A16) while
# eager derives with the activation's (2), so their divisibility tests differ.
# Either way eager would launch a row/col kernel over row-major bytes: right
# L2 norm, wrong direction, no exception. `_record_linear_layout` installs the
# guard only in that disagreeing case.
CPU_FALLBACK_PARTITION = -1

# Element width eager `aten::linear` derives its partition with. `rpu_linear`
# (src/ops/rpu_linear.cpp:501) uses `input.element_size()`, i.e. the ACTIVATION
# width, not the weight's — they differ for W8A16 (1 vs 2) and the activation
# is fp16 on every RPU path (`check_linear_shapes` rejects anything else).
EAGER_ACTIVATION_DWIDTH = 2


def _positive_int(
    value: object, name: str, *, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {parsed}")
    return parsed


def _validate_swizzle_tensor(
    weight: torch.Tensor, num_cores: object, dwidth: object
) -> tuple[int, int]:
    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            f"weight must be a torch.Tensor, got {type(weight).__name__}"
        )
    if weight.dim() != 2:
        raise ValueError(
            f"weight must be 2D [out_features, in_features], got "
            f"shape {tuple(weight.shape)}"
        )
    cores = _positive_int(num_cores, "num_cores", maximum=NUM_CORES)
    width = _positive_int(dwidth, "dwidth", maximum=32)
    if 32 % width != 0:
        raise ValueError(f"dwidth must divide 32 bytes, got {width}")
    actual_width = weight.element_size()
    if width != actual_width:
        raise ValueError(
            f"dwidth {width} does not match weight element size {actual_width}"
        )
    return cores, width


def get_linear_partition(in_features: int, out_features: int, dwidth: int = 2, num_cores: int = NUM_CORES) -> int:
    """
    Get partition strategy for given weight shape.

    Args:
        in_features: K dimension (weight.size(1))
        out_features: N dimension (weight.size(0))
        dwidth: element size in bytes (2 for fp16)
        num_cores: number of cores for partitioning (default NUM_CORES=8)

    Returns:
        0: row partition (按 K 维度切分)
        1: col partition (按 N 维度切分)
        -1: fallback to CPU (shapes not supported)
    """
    in_features = _positive_int(in_features, "in_features")
    out_features = _positive_int(out_features, "out_features")
    dwidth = _positive_int(dwidth, "dwidth", maximum=32)
    num_cores = _positive_int(num_cores, "num_cores", maximum=NUM_CORES)
    if 32 % dwidth != 0:
        raise ValueError(f"dwidth must divide 32 bytes, got {dwidth}")
    num_ele_32B = 32 // dwidth

    # Row partition 要求: K % (num_ele_32B * num_cores) == 0, N % 16 == 0
    can_row = (in_features % (num_ele_32B * num_cores) == 0) and (out_features % 16 == 0)

    # Col partition 要求: N % (16 * num_cores) == 0, K % num_ele_32B == 0
    can_col = (out_features % (16 * num_cores) == 0) and (in_features % num_ele_32B == 0)

    if can_row and can_col:
        # Fixed ABI contract: Python weight swizzle must match native
        # rpu_get_linear_partition(), which selects col when both are legal.
        return 1
    elif can_row:
        return 0
    elif can_col:
        return 1
    else:
        return -1


# D-04 export alias per F2(b) planner decision.
# feedback_linear_partition.md: partition scheme is FIXED, do not modify.
LinearPartition = get_linear_partition


def assert_fp16(tensor: torch.Tensor, name: str = "") -> None:
    """Raise if `tensor` is floating-point but not fp16. No-op for integer tensors."""
    if tensor.is_floating_point() and tensor.dtype != torch.float16:
        raise TypeError(
            f"{name or 'tensor'}: expected torch.float16, got {tensor.dtype}"
        )


def tp_row_swizzle_mc_weight(w: torch.Tensor, num_cores: int = NUM_CORES, dwidth: int = 2) -> torch.Tensor:
    """
    Row partition: 按 K 维度切分 weight

    Transform: (N, K) -> permuted (N, K) for row partition
    """
    num_cores, dwidth = _validate_swizzle_tensor(w, num_cores, dwidth)
    N, K = w.shape
    num_ele_32B = 32 // dwidth

    if N % 16 != 0:
        raise ValueError(f"N ({N}) must be divisible by 16 for row swizzle")
    k_alignment = num_ele_32B * num_cores
    if K % k_alignment != 0:
        raise ValueError(
            f"K ({K}) must be divisible by {k_alignment} for row swizzle"
        )

    # new shape = (N/16, 16, num_cores, K/num_ele_32B/num_cores, num_ele_32B)
    v = w.view(N // 16, 16, num_cores, K // num_ele_32B // num_cores, num_ele_32B)

    # permute(0, 3, 2, 1, 4) and reshape back
    p = v.permute(0, 3, 2, 1, 4).reshape(N, K)

    return p.contiguous()


def tp_col_swizzle_mc_weight(w: torch.Tensor, num_cores: int = NUM_CORES, dwidth: int = 2) -> torch.Tensor:
    """
    Col partition: 按 N 维度切分 weight

    Transform: (N, K) -> permuted (N, K) for col partition
    """
    num_cores, dwidth = _validate_swizzle_tensor(w, num_cores, dwidth)
    N, K = w.shape
    num_ele_32B = 32 // dwidth

    n_alignment = 16 * num_cores
    if N % n_alignment != 0:
        raise ValueError(
            f"N ({N}) must be divisible by {n_alignment} for col swizzle"
        )
    if K % num_ele_32B != 0:
        raise ValueError(
            f"K ({K}) must be divisible by {num_ele_32B} for col swizzle"
        )

    # new shape = (num_cores, N/16/num_cores, 16, K/num_ele_32B, num_ele_32B)
    v = w.view(num_cores, N // 16 // num_cores, 16, K // num_ele_32B, num_ele_32B)

    # permute(1, 3, 0, 2, 4) and reshape back
    p = v.permute(1, 3, 0, 2, 4).reshape(N, K)

    return p.contiguous()


def transform_linear_weight(weight: torch.Tensor, partition: int, num_cores: int = NUM_CORES) -> torch.Tensor:
    """
    Transform linear weight for RPU kernel.

    Args:
        weight: shape (out_features, in_features)
        partition: 0 for row, 1 for col
        num_cores: number of cores for partitioning (default NUM_CORES=8)

    Returns:
        Transformed weight with same shape but different layout
    """
    if isinstance(partition, bool) or not isinstance(partition, Integral):
        raise TypeError(f"partition must be an integer, got {partition!r}")
    partition = int(partition)
    if partition not in (0, 1):
        raise ValueError(f"partition must be 0 or 1, got {partition}")
    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            f"weight must be a torch.Tensor, got {type(weight).__name__}"
        )

    dwidth = weight.element_size()

    if partition == 0:
        return tp_row_swizzle_mc_weight(weight, num_cores, dwidth)
    else:
        return tp_col_swizzle_mc_weight(weight, num_cores, dwidth)


def _collect_embedding_data_ptrs(module: nn.Module) -> set:
    """Collect data_ptr() of all Embedding layer weights to detect tied weights."""
    embedding_ptrs = set()
    for m in module.modules():
        if isinstance(m, nn.Embedding):
            embedding_ptrs.add(m.weight.data_ptr())
    return embedding_ptrs


# ---------------------------------------------------------------------------
# Single-rule layout contract (2026-08-09)
# ---------------------------------------------------------------------------
# There used to be TWO independent rules deciding one thing — how a Linear's
# weight is partitioned:
#
#   A) this converter, which pins `o_proj`/`down_proj` to ROW *by name* and
#      swizzles attention projections with `attn_num_cores`, and
#   B) eager `aten::linear`, which re-derives the partition from the SHAPE
#      alone via `rpu_get_linear_partition` (src/ops/rpu_linear.cpp:419) and
#      whose DDR kernel hardcodes NUM_CORES.
#
# When both partitions are legal, the shape rule can take its col tie-break
# while the name rule takes row. `attn_num_cores != 8` (PaliGemma2
# num_kv_heads=4, Pi0.5 VLM=1) makes q/k/v/o_proj diverge the same way even
# when the partition ID matches, because the swizzle interleaves over a
# different core count. A weight swizzled under A and then consumed under B is
# read with the wrong layout: right L2 norm, wrong direction, no exception —
# the P1 double-swizzle failure mode (docs/pitfalls.md#p1).
#
# The fix is NOT to pick one of the two values — row is genuinely required by
# the fused SPM decoders (they launch o_proj/down_proj with partition=0) and
# col is genuinely required by the eager DDR kernel. It is to leave exactly ONE
# authority: this converter decides, records what it decided on the module, and
# the eager path honours that record instead of re-deriving its own answer.
# `_record_linear_layout` + `_guarded_linear_forward` are that honouring: the
# call is routed to the kernel the recorded layout actually needs, and hard-
# fails when the eager kernel structurally cannot serve it (num_cores !=
# NUM_CORES — the DDR kernel has no per-call core count, so "honour it" is not
# even expressible there).
#
# `SKIP_LINEAR_NAMES` is the third case of the same thing, and the one the
# original fix left open: the converter decides NOT to swizzle, so the recorded
# layout is "row-major, companion-owned" (`SKIPPED_PARTITION`). Eager dispatch
# cannot honour that either — a row-major weight is not a layout any partition
# reads — so it refuses. That refusal is the only reader the skip decision has;
# before 2026-08-09 the skip was written into the returned `partition_info` dict
# as `-2` and never read by anything, so a skipped Linear reaching eager
# `aten::linear` computed silent garbage.
def _eager_dispatch_layout(in_features: int, out_features: int) -> tuple[int, int]:
    """The (partition, num_cores) eager `aten::linear` assumes for this shape.

    Mirrors `rpu_get_linear_partition` (src/ops/rpu_linear.cpp:419) plus the
    hardcoded NUM_CORES in `rpu_launch_linear_ddr_kernel`. The element width is
    NOT a parameter on purpose: C++ passes `input.element_size()`
    (src/ops/rpu_linear.cpp:501), so this mirror must use the activation width
    too. Feeding it the weight's width (they differ under W8A16: 1 vs 2) makes
    this predict a partition eager will never choose, and the "eager re-derives
    the same answer, nothing to honour" early-return in `_record_linear_layout`
    is then computed against a fiction — skipping a guard that IS needed.
    """
    return (
        get_linear_partition(
            in_features, out_features, EAGER_ACTIVATION_DWIDTH,
            num_cores=NUM_CORES,
        ),
        NUM_CORES,
    )


def _check_linear_shapes(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> None:
    """Python mirror of `check_linear_shapes` (src/ops/rpu_linear.cpp:37).

    `_guarded_linear_forward` bypasses `aten::linear` -> `rpu_linear` and goes
    straight to `linear_with_partition`, whose only precondition is
    `input.dim() == 2` (src/core/rpu_pybind.inc:121). Everything `rpu_linear`
    checked first — fp16 activation above all — was silently dropped in that
    detour, so an fp32 rpu activation fed fp32 bytes to an fp16 DDR kernel and
    produced garbage with no exception.

    Note the fp16 rule is a HARD ERROR, not a CPU fallback: `rpu_linear` does
    have a `dtype != Half -> cpu_fallback_linear` branch (:494), but
    `check_linear_shapes` runs at :480 and TORCH_CHECKs Half first, so that
    branch is unreachable. This mirrors the behaviour, not the dead code.
    """
    if x.dtype != torch.float16:
        raise RuntimeError(
            f"rpu linear: only half supported, got {x.dtype}. (Mirrors "
            f"check_linear_shapes, src/ops/rpu_linear.cpp:37; the fp16 DDR "
            f"kernel would read the other dtype's bytes as fp16.)"
        )
    if weight.dim() != 2:
        raise RuntimeError(
            f"rpu linear: weight must be 2-D (out_features, in_features), got "
            f"{weight.dim()}-D"
        )
    if x.dim() < 1:
        raise RuntimeError(
            f"rpu linear: input must have at least 1 dim, got {x.dim()}-D"
        )
    if x.size(-1) != weight.size(1):
        raise RuntimeError(
            f"rpu linear: last dimension of input ({x.size(-1)}) does not "
            f"match weight.size(1) (in_features={weight.size(1)})"
        )
    if bias is not None:
        if bias.dim() > 1:
            raise RuntimeError(
                f"rpu linear: bias must be 1-D or scalar, got {bias.dim()}-D"
            )
        if bias.dim() == 1 and bias.size(0) != weight.size(0):
            raise RuntimeError(
                f"rpu linear: bias length ({bias.size(0)}) must equal "
                f"out_features ({weight.size(0)})"
            )


def _guarded_linear_forward(
    self: nn.Linear, x: torch.Tensor, *, recorded_layout: tuple[int, int | None],
) -> torch.Tensor:
    """`nn.Linear.forward` that honours the recorded swizzle layout.

    Installed only on Linears whose recorded layout differs from what eager
    `aten::linear` would re-derive. Off-device calls keep stock behaviour.
    """
    if x.device.type != "rpu":
        return torch.nn.functional.linear(x, self.weight, self.bias)

    # Bind the converter's decision, including on owners without an A9 stamp.
    # Mutable attributes remain inspectable mirrors, not layout authority.
    for name, expected in zip(
        ("_rpu_linear_partition", "_rpu_linear_num_cores"), recorded_layout,
    ):
        actual = vars(self).get(name)
        if name not in vars(self) or type(actual) is not type(expected) or actual != expected:
            raise RuntimeError(f"rpu linear: recorded swizzle layout changed: {name}")
    partition, num_cores = recorded_layout
    if partition == CPU_FALLBACK_PARTITION:
        raise RuntimeError(
            "rpu linear: get_linear_partition refused this shape "
            f"(in_features={self.in_features}, out_features={self.out_features}"
            f", weight dwidth={self.weight.element_size()}), so the converter "
            "left the weight ROW-MAJOR and expected a CPU fallback. Eager "
            "aten::linear re-derives the partition from the shape alone with "
            f"num_cores={NUM_CORES} and the fp16 ACTIVATION width, gets a legal "
            "row/col answer here, and would launch a swizzled-layout kernel "
            "over row-major bytes: right L2 norm, wrong direction, no "
            "exception (docs/pitfalls.md#p1). Run this module on CPU."
        )
    if partition == SKIPPED_PARTITION:
        raise RuntimeError(
            "rpu linear: this Linear is in SKIP_LINEAR_NAMES, so the converter "
            "left its weight ROW-MAJOR on purpose and a companion swizzler owns "
            "the RPU copy (AdaRMS dense/cond expansion, the Qwen3.5 q/k/v "
            "re-swizzle, the Pi0.5 int4 packer). Eager aten::linear assumes a "
            "swizzled weight, so calling this module on an rpu tensor reads "
            "row-major bytes as swizzled ones: right L2 norm, wrong direction, "
            "no exception (docs/pitfalls.md#p1). Run the fused forward that "
            "consumes the companion's copy, or run this module on CPU."
        )
    _check_linear_shapes(x, self.weight, self.bias)

    # The C-ext symbols live on `_cpp_ext`, not on the package namespace (the
    # v5 cutover whitelist keeps `vars(rpu_backend)` closed).
    from rpu_backend import _cpp_ext

    x2d = x.reshape(-1, x.shape[-1]).contiguous() if x.dim() != 2 else x.contiguous()
    if num_cores != NUM_CORES:
        out = torch.ops.rpu.linear_with_core_count(
            x2d, self.weight, self.bias, partition, num_cores)
    else:
        out = _cpp_ext.linear_with_partition(x2d, self.weight, self.bias, partition)
    if x.dim() == 2:
        return out
    return out.reshape(*x.shape[:-1], out.shape[-1])


def _record_linear_layout(
    linear: nn.Linear, partition: int, num_cores: int | None
) -> None:
    """Record the chosen layout, and make eager dispatch honour it.

    `partition=SKIPPED_PARTITION` records "this converter did not swizzle it"
    and `partition=CPU_FALLBACK_PARTITION` records "no legal partition at the
    core count / element width I used"; in both cases no swizzle happened, so
    `num_cores` is meaningless and is recorded None.
    """
    linear._rpu_linear_partition = partition
    linear._rpu_linear_num_cores = num_cores

    if partition != SKIPPED_PARTITION:
        eager = _eager_dispatch_layout(linear.in_features, linear.out_features)
        if partition == CPU_FALLBACK_PARTITION:
            if eager[0] == CPU_FALLBACK_PARTITION:
                # Eager refuses the shape too, so `rpu_linear` warns once and
                # CPU-falls-back — the right answer for a row-major weight.
                return
        elif (partition, num_cores) == eager:
            return  # eager re-derives the same answer; nothing to honour
    linear.forward = types.MethodType(
        partial(_guarded_linear_forward, recorded_layout=(partition, num_cores)), linear,
    )


def validate_decoder_weight_structure(model, *, num_layers, hidden_size,
        intermediate_size, num_q_heads, num_kv_heads, head_dim, vocab_size,
        projection_dtype=torch.float16, lm_head_dtype=torch.float16):
    """Validate the actual plain decoder before irreversible weight conversion.

    HF config describes intended geometry; modules may have been replaced after
    loading. Check every physical weight, including the last layer and the eager
    vocabulary projection, before claiming a device model or publishing markers.
    """
    if any(dtype not in (torch.float16, torch.int8)
           for dtype in (projection_dtype, lm_head_dtype)):
        raise ValueError("decoder structure admits FP16 or per-channel INT8 projections")

    def weight(owner, name, shape, dtype=torch.float16):
        value = getattr(owner, "weight", None)
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype != dtype):
            label = "INT8" if dtype == torch.int8 else "FP16"
            raise ValueError(f"decoder {name} requires {label} weight shape {tuple(shape)}")
        if getattr(owner, "bias", None) is not None:
            raise ValueError(f"decoder {name} does not admit a bias")

    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", ())
    if len(layers) != num_layers:
        raise ValueError(f"decoder requires exactly {num_layers} actual layers, got {len(layers)}")
    if not isinstance(getattr(decoder, "embed_tokens", None), nn.Embedding):
        raise ValueError("decoder embedding must be an Embedding")
    if not isinstance(getattr(model, "lm_head", None), nn.Linear):
        raise ValueError("decoder lm_head must be a Linear")
    weight(getattr(decoder, "embed_tokens", None), "embedding", (vocab_size, hidden_size))
    weight(getattr(model, "lm_head", None), "lm_head", (vocab_size, hidden_size), lm_head_dtype)
    weight(getattr(decoder, "norm", None), "final norm", (hidden_size,))
    shapes = {"q_proj": (num_q_heads * head_dim, hidden_size),
        "k_proj": (num_kv_heads * head_dim, hidden_size),
        "v_proj": (num_kv_heads * head_dim, hidden_size),
        "o_proj": (hidden_size, num_q_heads * head_dim),
        "gate_proj": (intermediate_size, hidden_size),
        "up_proj": (intermediate_size, hidden_size),
        "down_proj": (hidden_size, intermediate_size)}
    for index, layer in enumerate(layers):
        attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
        if (getattr(attn, "head_dim", None) != head_dim
                or not getattr(attn, "is_causal", True)
                or getattr(attn, "sliding_window", None) is not None):
            raise ValueError(f"decoder layer {index} requires the plain causal attention geometry")
        for name, shape in shapes.items():
            projection = getattr(attn if name in {"q_proj", "k_proj", "v_proj", "o_proj"} else mlp, name, None)
            if not isinstance(projection, nn.Linear):
                raise ValueError(f"decoder layer {index} {name} must be a Linear")
            if (projection.out_features, projection.in_features) != shape:
                raise ValueError(f"decoder layer {index} {name} declared shape differs from {shape}")
            weight(projection, f"layer {index} {name}", shape, projection_dtype)
        for owner, name, shape in ((layer, "input_layernorm", (hidden_size,)),
                (layer, "post_attention_layernorm", (hidden_size,)),
                (attn, "q_norm", (head_dim,)), (attn, "k_norm", (head_dim,))):
            weight(getattr(owner, name, None), f"layer {index} {name}", shape)

    known = {"lm_head"}
    for index in range(num_layers):
        known.update(f"model.layers.{index}.self_attn.{name}" for name in
                     ("q_proj", "k_proj", "v_proj", "o_proj"))
        known.update(f"model.layers.{index}.mlp.{name}" for name in
                     ("gate_proj", "up_proj", "down_proj"))
    for path, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, nn.Linear) and path not in known:
            raise ValueError(f"decoder has unknown Linear role: {path!r}")


def pad_decoder_mlp_projection(weight, *, projection_name: str,
                               logical_size: int, physical_size: int):
    """Prepare one original CPU FP16 projection before its single swizzle.

    Adapters validate all original leaves first. This returns a cold copy so
    fused adapters can retain logical HF metadata and discard each raw leaf.
    """
    if (logical_size, physical_size) not in ((3584, 3648), (9728, 9792)):
        raise ValueError("no admitted decoder MLP padding geometry")
    if projection_name not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError("MLP padding requires a known projection role")
    axis = 1 if projection_name == "down_proj" else 0
    if (weight.device.type != "cpu" or weight.dtype != torch.float16
            or weight.ndim != 2 or not weight.is_contiguous()
            or weight.shape[axis] != logical_size):
        raise ValueError("MLP padding requires original contiguous CPU FP16 weights")
    shape = list(weight.shape)
    shape[axis] = physical_size
    with torch.no_grad():
        padded = weight.new_zeros(shape)
        padded[:weight.shape[0], :weight.shape[1]].copy_(weight)
    return padded


def validate_decoder_mlp_padding(model, *, logical_size: int, physical_size: int):
    """Check every affected original leaf before claiming or mutating a model."""
    if physical_size == logical_size:
        return
    if (logical_size, physical_size) not in ((3584, 3648), (9728, 9792)):
        raise ValueError("no admitted decoder MLP padding geometry")
    layers = model.model.layers
    hidden = model.config.hidden_size
    for layer in layers:
        for name, shape in (("gate_proj", (logical_size, hidden)),
                            ("up_proj", (logical_size, hidden)),
                            ("down_proj", (hidden, logical_size))):
            projection = getattr(layer.mlp, name)
            w = projection.weight
            if (w.device.type != "cpu" or w.dtype != torch.float16
                    or not w.is_contiguous() or tuple(w.shape) != shape
                    or (projection.out_features, projection.in_features) != shape
                    or projection.bias is not None
                    or hasattr(projection, "_rpu_linear_partition")):
                raise ValueError("MLP padding requires original contiguous CPU FP16 weights")


def pad_decoder_mlp_weights(model, *, logical_size: int, physical_size: int):
    """Pad CPU weights once before swizzle; forward needs no pad/slice.

    Config stays logical; Linear shapes and their recorded partition become
    physical. A repeated invocation refuses already converted weights.
    """
    validate_decoder_mlp_padding(model, logical_size=logical_size, physical_size=physical_size)
    if physical_size == logical_size:
        return
    layers = model.model.layers
    hidden = model.config.hidden_size
    with torch.no_grad():
        for layer in layers:
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection = getattr(layer.mlp, name)
                w = projection.weight
                padded = pad_decoder_mlp_projection(w, projection_name=name,
                    logical_size=logical_size, physical_size=physical_size)
                projection.weight = nn.Parameter(padded, requires_grad=w.requires_grad)
                projection.out_features, projection.in_features = padded.shape
    _release_cpu_weight_pages()


def _linear_conversion_layout(linear, name, force_col_partition,
                              attn_num_cores, mlp_num_cores, lm_head_num_cores):
    cores = (attn_num_cores if name in {"q_proj", "k_proj", "v_proj", "o_proj"}
             else mlp_num_cores if name in {"gate_proj", "up_proj", "down_proj"}
             else lm_head_num_cores if name == "lm_head" else NUM_CORES)
    if not force_col_partition and name in ROW_PARTITION_NAMES:
        return 0, cores
    partition = get_linear_partition(linear.in_features, linear.out_features,
                                     linear.weight.element_size(), num_cores=cores)
    return (-1 if force_col_partition and partition != 1 else partition), cores


def _validate_bounded_linear_move(module, force_col_partition, attn_num_cores,
                                  mlp_num_cores, lm_head_num_cores, skip_names):
    # Validate the whole tree before the first move, including a bad last leaf.
    # Do not retain a Parameter/state_dict plan: Module.to may replace those
    # Parameters, and retaining them would keep the complete CPU model alive.
    roles = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
             "down_proj", "lm_head"}
    for path, child in module.named_modules():
        if isinstance(child, nn.Embedding):
            weight = child.weight
            if (weight.device.type != "cpu" or weight.dtype != torch.float16
                    or not weight.is_contiguous()
                    or tuple(weight.shape) != (child.num_embeddings, child.embedding_dim)):
                raise ValueError(f"bounded weight migration requires a contiguous CPU FP16 Embedding: {path!r}")
            continue
        if not isinstance(child, nn.Linear):
            continue
        name = path.rsplit(".", 1)[-1]
        if name not in roles or name in skip_names:
            raise ValueError(f"bounded weight migration has unknown or skipped Linear role: {path!r}")
        weight = child.weight
        if (weight.device.type != "cpu" or weight.dtype != torch.float16
                or not weight.is_contiguous()
                or tuple(weight.shape) != (child.out_features, child.in_features)
                or hasattr(child, "_rpu_linear_partition")):
            raise ValueError(f"bounded weight migration requires a contiguous CPU FP16 Linear: {path!r}")
        partition, cores = _linear_conversion_layout(child, name, force_col_partition,
            attn_num_cores, mlp_num_cores, lm_head_num_cores)
        n, k = weight.shape
        if ((partition == 0 and (n % 16 or k % (16 * cores)))
                or (partition == 1 and (n % (16 * cores) or k % 16))
                or partition < 0):
            raise ValueError(f"bounded weight migration has no legal Linear layout: {path!r}")
        if child.bias is not None and (child.bias.device.type != "cpu"
                or child.bias.dtype != torch.float16
                or tuple(child.bias.shape) != (child.out_features,)):
            raise ValueError(f"bounded weight migration requires a CPU FP16 Linear bias: {path!r}")


def _release_cpu_weight_pages():
    # RPU HostDDR shares host memory. Return freed CPU arenas before the next
    # device allocation instead of retaining a second model-sized RSS peak.
    import ctypes
    import gc
    gc.collect()
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    trim(0)


def convert_linear_weights_inplace(module: nn.Module, prefix: str = "", _embedding_ptrs: set = None, force_col_partition: bool = False, attn_num_cores: int = NUM_CORES, skip_names: set = None, *, mlp_num_cores: int = NUM_CORES, lm_head_num_cores: int = NUM_CORES, execution_core_count: int = NUM_CORES, move_to_device=None) -> Dict[str, int]:
    """
    Recursively convert all Linear layer weights in a module to RPU layout.

    Args:
        module: PyTorch module to convert
        prefix: Parameter name prefix for logging
        _embedding_ptrs: (internal) Set of embedding weight data_ptr() for tied weight detection
        force_col_partition: If True, always use col partition (for DDR linear without fused decoder)
        attn_num_cores: Number of cores for attention layers (q/k/v/o_proj).
                        Use min(NUM_CORES, num_kv_heads) for models with fewer KV heads than cores.
                        MLP layers (gate/up/down_proj) always use NUM_CORES.
        skip_names: Additional Linear child names to skip (merged with built-in {'dense'}).
                    Use skip_names={'cond'} for QwenPI05 AdaRMS conditioning Linears.
                    A skipped Linear keeps its row-major weight and is recorded
                    `SKIPPED_PARTITION`; calling it eagerly on an rpu tensor then
                    raises, because only its companion swizzler knows the layout.
        move_to_device: Internal cold-install option. Validate all original CPU
            FP16 weights, reserve the vocabulary head and embedding first, then
            convert and move one Linear at a time. Default: no device migration.

    Returns:
        Dict mapping parameter names to their partition strategy
        (0 row / 1 col / -1 CPU fallback / SKIPPED_PARTITION skipped)
    """
    attn_num_cores = _positive_int(attn_num_cores, "attn_num_cores", maximum=NUM_CORES)
    mlp_num_cores = _positive_int(mlp_num_cores, "mlp_num_cores", maximum=NUM_CORES)
    lm_head_num_cores = _positive_int(lm_head_num_cores, "lm_head_num_cores", maximum=NUM_CORES)
    # A leaf `nn.Linear` has no `named_children()`, so the walk below would
    # return `{}` having converted nothing — silently leaving a row-major
    # weight that the RPU kernel then reads as swizzled. That silent no-op has
    # already faked a positive result twice (pi05 CR-R5 BLOCKER-1, see
    # adapters/pi05/weights.py:19-31; and tests/ops/test_ops_individual.py's
    # Linear-col leg). Refuse instead of returning an empty dict.
    if isinstance(module, nn.Linear):
        raise TypeError(
            "convert_linear_weights_inplace walks named_children(); a bare "
            "nn.Linear has none, so this call would convert nothing and "
            "return {}. Wrap it in its parent module, or swizzle the leaf "
            "directly with transform_linear_weight(w, get_linear_partition(...))."
        )

    execution_core_count = _positive_int(execution_core_count, "execution_core_count", maximum=NUM_CORES)
    if move_to_device is not None:
        move_to_device = torch.device(move_to_device)
        if _embedding_ptrs is None:
            _validate_bounded_linear_move(module, force_col_partition,
                attn_num_cores, mlp_num_cores, lm_head_num_cores,
                {'dense'} if skip_names is None else skip_names | {'dense'})
    # Inspect the complete tree before irreversible conversion. A reduced budget
    # has no unmodelled Linear fallback, even if that child appears last.
    if _embedding_ptrs is None and execution_core_count != NUM_CORES:
        roles = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"}
        # Specialized owners explicitly skip their projections and prepare them
        # separately. They must remain raw, as in the eight-core conversion.
        roles |= {'dense'} if skip_names is None else (skip_names | {'dense'})
        for path, child in module.named_modules():
            if isinstance(child, nn.Linear) and path.rsplit(".", 1)[-1] not in roles:
                raise ValueError(f"reduced-core weight conversion has unknown Linear role: {path!r}")
        if any(cores > execution_core_count for cores in (attn_num_cores, mlp_num_cores, lm_head_num_cores)):
            raise ValueError("Linear role core count exceeds the model execution budget")
    # On first call, collect embedding weight pointers to detect tied weights
    if _embedding_ptrs is None:
        _embedding_ptrs = _collect_embedding_data_ptrs(module)

    # Linear children whose weights must NOT be swizzled by this pass.
    # `PiGemmaRMSNorm.dense` (lerobot/policies/pi_gemma.py) is an AdaRMS
    # scale/shift/gate projection that needs its own RPU-specific expansion
    # (`.repeat(NUM_CORES, 1)` + col-partition swizzle) in
    # `adarms_converter._prepare_adarms_dense_weights`. Pre-swizzling it here
    # corrupts the downstream AdaRMS GEMV path (double-swizzle → wrong
    # scale/shift/gate per layer, same magnitude but scrambled direction).
    SKIP_LINEAR_NAMES = {'dense'} if skip_names is None else (skip_names | {'dense'})

    partition_info = {}

    children = module.named_children()
    if move_to_device is not None:
        # Snapshot embedding pointers before every move. Convert a tied head
        # before its embedding, then allocate both large vocabulary tensors
        # before many smaller live layer allocations fragment HostDDR.
        # Retain module references only; old Parameters must be released.
        children = sorted(children, key=lambda item:
            0 if item[0] == "lm_head" and isinstance(item[1], nn.Linear)
            else 1 if isinstance(item[1], nn.Embedding) else 2)
    for name, child in children:
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            if name in SKIP_LINEAR_NAMES:
                # AdaRMS dense: consumed by adarms_converter with its own swizzle.
                # "Skipped" is not "no layout" — the weight stays row-major while
                # a companion swizzler owns the RPU copy, and every in-tree
                # companion builds that copy SEPARATELY (adarms `_rpu_dense_w_rp`,
                # qwenpi05/rhinovla `_rpu_cond_w_rp`, the Qwen3.5 q/k/v lists
                # handed to C++). So the module is left holding bytes eager
                # `aten::linear` would misread. Record the skip on the module so
                # dispatch refuses instead of computing on them.
                _record_linear_layout(child, SKIPPED_PARTITION, num_cores=None)
                partition_info[f"{full_name}.weight"] = SKIPPED_PARTITION
                continue

            assert_fp16(child.weight, f"{full_name}.weight")
            partition, layer_num_cores = _linear_conversion_layout(
                child, name, force_col_partition, attn_num_cores,
                mlp_num_cores, lm_head_num_cores)

            # Check if this Linear layer shares weights with an Embedding layer (tied weights)
            is_tied = child.weight.data_ptr() in _embedding_ptrs

            if partition >= 0:
                with torch.no_grad():
                    if is_tied:
                        # Untie the weights: clone before transforming so embedding is not affected
                        child.weight = nn.Parameter(child.weight.data.clone())

                    # Transform weight
                    transformed = transform_linear_weight(child.weight.data, partition, num_cores=layer_num_cores)
                    child.weight.data = transformed
                    del transformed

                # Single-rule contract: record what we just chose so eager
                # dispatch honours it instead of re-deriving its own answer.
                _record_linear_layout(child, partition, layer_num_cores)
                partition_info[f"{full_name}.weight"] = partition
            else:
                # Same single-rule contract as the two branches above: the
                # converter decided NOT to swizzle (no legal partition at
                # `layer_num_cores` / this weight's dwidth, or force_col
                # demoted a row answer), so record that decision. Without this
                # the module carries no `_rpu_linear_partition` at all and
                # eager `aten::linear` silently re-derives its own — which can
                # be 0 or 1 where this said -1.
                _record_linear_layout(
                    child, CPU_FALLBACK_PARTITION, num_cores=None
                )
                partition_info[f"{full_name}.weight"] = CPU_FALLBACK_PARTITION
            if move_to_device is not None:
                child.to(move_to_device)
                _release_cpu_weight_pages()
        elif isinstance(child, nn.Embedding) and move_to_device is not None:
            child.to(move_to_device)
            _release_cpu_weight_pages()
        else:
            # Recurse into child modules
            child_info = convert_linear_weights_inplace(child, full_name, _embedding_ptrs, force_col_partition, attn_num_cores, skip_names,
                mlp_num_cores=mlp_num_cores, lm_head_num_cores=lm_head_num_cores,
                execution_core_count=execution_core_count, move_to_device=move_to_device)
            partition_info.update(child_info)

    return partition_info


# D-03 canonical names. Aliases preserve back-compat for sibling converters
# importing the legacy names through `model_converter.py`.
swizzle_linear = transform_linear_weight  # D-03 canonical name; transform_linear_weight kept as legacy alias
swizzle_model_inplace = convert_linear_weights_inplace  # D-03 canonical name; convert_linear_weights_inplace kept as legacy alias (deprecated per §11, v5.0 tombstone)


# Legacy entry points lifted from _internal/weights.py
def convert_model_for_rpu(
    model: nn.Module,
    save_path: str,
    device: str = "cpu",
    half: bool = True,
) -> Dict[str, int]:
    """Convert a PyTorch model for RPU inference and save it.

    This function:
    1. Converts model to fp16 (if half=True)
    2. Transforms all Linear layer weights to RPU-optimized layout
    3. Saves the converted model

    Returns:
        Dict mapping parameter names to their partition strategy
    """
    if not isinstance(half, bool):
        raise TypeError(f"half must be a bool, got {half!r}")

    # Make a deep copy to avoid modifying the original
    model_copy = copy.deepcopy(model)
    model_copy = model_copy.to(device)

    if half:
        model_copy = model_copy.half()

    partition_info = convert_linear_weights_inplace(model_copy)

    save_data = {
        "state_dict": model_copy.state_dict(),
        "partition_info": partition_info,
        "rpu_converted": True,
    }
    torch.save(save_data, save_path)

    total_layers = len(partition_info)
    num_row = sum(1 for v in partition_info.values() if v == 0)
    num_col = sum(1 for v in partition_info.values() if v == 1)
    num_converted = num_row + num_col
    num_fallback = sum(1 for v in partition_info.values() if v < 0)
    _LOG.info("Converted %d/%d Linear layers (col: %d, row: %d, fallback: %d), saved to %s",
              num_converted, total_layers, num_col, num_row, num_fallback, save_path)
    return partition_info


def load_rpu_model(
    model: nn.Module,
    load_path: str,
    device: str = "rpu",
    strict: bool = True,
) -> nn.Module:
    """Load a model that was converted for RPU inference."""
    if not isinstance(strict, bool):
        raise TypeError(f"strict must be a bool, got {strict!r}")
    save_data = torch.load(
        load_path, map_location="cpu", weights_only=True
    )
    if not isinstance(save_data, Mapping):
        raise TypeError(
            "converted RPU model artifact must contain a mapping, got "
            f"{type(save_data).__name__}"
        )

    if save_data.get("rpu_converted") is not True:
        raise ValueError(
            "model artifact is not marked as converted for RPU; refusing to "
            "load unswizzled weights"
        )

    state_dict = save_data.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("converted RPU model artifact requires a state_dict mapping")

    partition_info = None
    if "partition_info" in save_data:
        partition_info = save_data["partition_info"]
        if not isinstance(partition_info, Mapping):
            raise TypeError(
                "converted RPU model artifact has invalid partition_info"
            )
        for name, value in partition_info.items():
            if (
                not isinstance(name, str)
                or isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) not in (-2, -1, 0, 1)
            ):
                raise TypeError(
                    "converted RPU model artifact has invalid partition_info"
                )

    model.load_state_dict(state_dict, strict=strict)
    if partition_info is not None:
        num_converted = sum(1 for v in partition_info.values() if v >= 0)
        _LOG.info("Loaded model with %d RPU-optimized Linear layers", num_converted)

    model = model.to(device)
    _LOG.info("Model moved to %s", device)
    return model


def check_model_rpu_compatible(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Check if a model's Linear layers are compatible with RPU acceleration."""
    info: Dict[str, Dict[str, Any]] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            dwidth = 2  # Assume fp16
            partition = get_linear_partition(in_features, out_features, dwidth)
            info[name] = {
                "in_features": in_features,
                "out_features": out_features,
                "partition": partition,
                "status": "row" if partition == 0 else ("col" if partition == 1 else "fallback"),
            }
    return info


def print_rpu_compatibility_report(model: nn.Module) -> None:
    """Print a compatibility report for a model."""
    info = check_model_rpu_compatible(model)
    num_converted = sum(1 for d in info.values() if d['partition'] >= 0)
    num_fallback = sum(1 for d in info.values() if d['partition'] < 0)
    num_row = sum(1 for d in info.values() if d['partition'] == 0)
    num_col = sum(1 for d in info.values() if d['partition'] == 1)
    _LOG.info("Compatibility: %d Linear layers (accelerated: %d, col: %d, row: %d, fallback: %d)",
              len(info), num_converted, num_col, num_row, num_fallback)


def validate_qwen3_vl_text_semantics(text_model) -> None:
    """Validate live text math before installing a prepacked VL profile."""
    from transformers.activations import SiLUActivation

    config = text_model.config
    for index, layer in enumerate(text_model.layers):
        activation = getattr(layer.mlp, "act_fn", None)
        if not (type(activation) is SiLUActivation or (
                type(activation) is nn.SiLU and activation.inplace is False)):
            raise ValueError(f"Qwen3-VL layer {index} requires exact SiLU activation")
        for owner, name in ((layer.self_attn, "q_norm"), (layer.self_attn, "k_norm"),
                            (layer, "input_layernorm"), (layer, "post_attention_layernorm")):
            if getattr(getattr(owner, name, None), "variance_epsilon", None) != 1e-6:
                raise ValueError(f"Qwen3-VL layer {index} {name} requires epsilon 1e-6")
    if getattr(text_model.norm, "variance_epsilon", None) != 1e-6:
        raise ValueError("Qwen3-VL final norm requires epsilon 1e-6")

    rotary = getattr(text_model, "rotary_emb", None)
    section = getattr(rotary, "mrope_section", None)
    rotary_config = getattr(rotary, "config", None)
    geometry = ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size")
    rope = (getattr(config, "rope_parameters", None)
            or getattr(config, "rope_scaling", None) or {})
    rotary_rope = (getattr(rotary_config, "rope_parameters", None)
                   or getattr(rotary_config, "rope_scaling", None) or {})
    if (getattr(rotary, "rope_type", None) != "default"
            or getattr(rotary, "attention_scaling", None) != 1.0
            or not isinstance(section, (list, tuple))
            or tuple(section) != (24, 20, 20)
            or any(isinstance(item, bool) or not isinstance(item, Integral) for item in section)
            or rotary_config is None
            or any(getattr(rotary_config, field, None) != getattr(config, field, None)
                   for field in geometry)
            or not isinstance(rope, Mapping) or not isinstance(rotary_rope, Mapping)
            or rotary_rope != rope):
        raise ValueError("Qwen3-VL requires exact live default interleaved M-RoPE semantics")
