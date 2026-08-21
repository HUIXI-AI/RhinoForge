"""Weight partitioning, layout transformation, and conversion helpers.

``SKIP_LINEAR_NAMES = {'dense'}`` protects AdaRMS projections that have an
authoritative component-specific transformation. See
``docs/model_porting.md#3-reuse-the-causal-decoder-when-it-matches``.
"""
from __future__ import annotations

import copy
import types
from collections.abc import Mapping
from numbers import Integral
from typing import Dict, Any

import torch
import torch.nn as nn

from rpu_backend.runtime.log import _LOG


NUM_CORES = 8

# Projection child names pinned to ROW partition regardless of shape, because
# that is what the fused SPM decoders launch them with. ``adapters/pi05/w4pack.py``
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
# uses `input.element_size()`, i.e. the activation
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


# Public alias for the fixed partition rule.
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
# Single-rule layout contract
# ---------------------------------------------------------------------------
# The converter is the layout authority: it records its partition and core
# count on each Linear, and eager dispatch honours that record instead of
# deriving a potentially different answer from shape alone.
# `_record_linear_layout` + `_guarded_linear_forward` are that honouring: the
# call is routed to the kernel the recorded layout actually needs, and hard-
# fails when the eager kernel structurally cannot serve it (num_cores !=
# NUM_CORES — the DDR kernel has no per-call core count, so "honour it" is not
# even expressible there).
#
# `SKIP_LINEAR_NAMES` is the third case: the converter does not swizzle, so the recorded
# layout is "row-major, companion-owned" (`SKIPPED_PARTITION`). Eager dispatch
# cannot honour that either, so it refuses instead of interpreting row-major
# bytes as a transformed layout.
def _eager_dispatch_layout(in_features: int, out_features: int) -> tuple[int, int]:
    """The (partition, num_cores) eager `aten::linear` assumes for this shape.

    Mirrors ``rpu_get_linear_partition`` plus the fixed ``NUM_CORES`` in the
    eager launch path. The element width is
    NOT a parameter on purpose: C++ passes `input.element_size()`
    so this mirror must use the activation width too. Feeding it the weight's
    width (they differ under W8A16: 1 vs 2) makes
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
    """Python mirror of the native eager-linear shape checks.

    `_guarded_linear_forward` bypasses `aten::linear` -> `rpu_linear` and goes
    straight to `linear_with_partition`, whose only precondition is
    `input.dim() == 2`. Everything `rpu_linear`
    checked first — fp16 activation above all — was silently dropped in that
    detour, so an fp32 rpu activation fed fp32 bytes to an fp16 DDR kernel and
    produced garbage with no exception.

    Note the fp16 rule is a HARD ERROR, not a CPU fallback: `rpu_linear` does
    contains a dtype fallback branch, but its shape check rejects non-FP16
    input first. This mirrors the reachable behavior.
    """
    if x.dtype != torch.float16:
        raise RuntimeError(
            f"rpu linear: only half supported, got {x.dtype}; the fp16 path "
            "cannot consume another dtype's bytes."
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


def _guarded_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """`nn.Linear.forward` that honours the recorded swizzle layout.

    Installed only on Linears whose recorded layout differs from what eager
    `aten::linear` would re-derive. Off-device calls keep stock behaviour.
    """
    if x.device.type != "rpu":
        return torch.nn.functional.linear(x, self.weight, self.bias)

    partition = self._rpu_linear_partition
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
            "exception. Run this module on CPU. See "
            "docs/model_porting.md#3-reuse-the-causal-decoder-when-it-matches."
        )
    if partition == SKIPPED_PARTITION:
        raise RuntimeError(
            "rpu linear: this Linear is in SKIP_LINEAR_NAMES, so the converter "
            "left its weight ROW-MAJOR on purpose and a companion swizzler owns "
            "the RPU copy (AdaRMS dense/cond expansion, the Qwen3.5 q/k/v "
            "re-swizzle, the Pi0.5 int4 packer). Eager aten::linear assumes a "
            "swizzled weight, so calling this module on an rpu tensor reads "
            "row-major bytes as swizzled ones: right L2 norm, wrong direction, "
            "no exception. Run the fused forward that "
            "consumes the companion's copy, or run this module on CPU."
        )
    num_cores = self._rpu_linear_num_cores
    if num_cores != NUM_CORES:
        raise RuntimeError(
            f"rpu linear: this weight was swizzled for {num_cores} cores "
            f"(attn_num_cores), but the eager aten::linear DDR kernel is fixed "
            f"at {NUM_CORES}. It has no per-call core count, so it cannot read "
            f"this layout. This Linear is a fused-subsystem weight — run it "
            f"through its fused forward, not as an eager module call."
        )

    _check_linear_shapes(x, self.weight, self.bias)

    # C-extension symbols live on `_cpp_ext`; the package namespace exposes
    # only the documented public names.
    from rpu_backend import _cpp_ext

    x2d = x.reshape(-1, x.shape[-1]).contiguous() if x.dim() != 2 else x.contiguous()
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
    linear.forward = types.MethodType(_guarded_linear_forward, linear)


def convert_linear_weights_inplace(module: nn.Module, prefix: str = "", _embedding_ptrs: set = None, force_col_partition: bool = False, attn_num_cores: int = NUM_CORES, skip_names: set = None) -> Dict[str, int]:
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

    Returns:
        Dict mapping parameter names to their partition strategy
        (0 row / 1 col / -1 CPU fallback / SKIPPED_PARTITION skipped)
    """
    # A leaf `nn.Linear` has no `named_children()`, so the walk below would
    # return `{}` having converted nothing — silently leaving a row-major
    # weight that the RPU kernel then reads as swizzled. That silent no-op has
    # leave row-major bytes that an RPU path could misread. Refuse instead of
    # returning an empty dict.
    if isinstance(module, nn.Linear):
        raise TypeError(
            "convert_linear_weights_inplace walks named_children(); a bare "
            "nn.Linear has none, so this call would convert nothing and "
            "return {}. Wrap it in its parent module, or swizzle the leaf "
            "directly with transform_linear_weight(w, get_linear_partition(...))."
        )

    # On first call, collect embedding weight pointers to detect tied weights
    if _embedding_ptrs is None:
        _embedding_ptrs = _collect_embedding_data_ptrs(module)

    # Attention layer names that should use attn_num_cores
    ATTN_LAYER_NAMES = {'q_proj', 'k_proj', 'v_proj', 'o_proj'}

    # Linear children whose weights must NOT be swizzled by this pass.
    # `PiGemmaRMSNorm.dense` (lerobot/policies/pi_gemma.py) is an AdaRMS
    # scale/shift/gate projection that needs its own RPU-specific expansion
    # (`.repeat(NUM_CORES, 1)` + col-partition swizzle) in
    # `adarms_converter._prepare_adarms_dense_weights`. Pre-swizzling it here
    # corrupts the downstream AdaRMS GEMV path (double-swizzle → wrong
    # scale/shift/gate per layer, same magnitude but scrambled direction).
    SKIP_LINEAR_NAMES = {'dense'} if skip_names is None else (skip_names | {'dense'})

    partition_info = {}

    for name, child in module.named_children():
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

            in_features = child.in_features
            out_features = child.out_features
            assert_fp16(child.weight, f"{full_name}.weight")
            dwidth = child.weight.element_size()

            # Determine num_cores for this layer
            is_attn_layer = name in ATTN_LAYER_NAMES
            layer_num_cores = attn_num_cores if is_attn_layer else NUM_CORES

            # Check if this Linear layer shares weights with an Embedding layer (tied weights)
            is_tied = child.weight.data_ptr() in _embedding_ptrs

            if force_col_partition:
                # DDR linear mode: all layers use col partition (no cross-core reduce-sum)
                partition = get_linear_partition(
                    in_features, out_features, dwidth,
                    num_cores=layer_num_cores,
                )
                if partition != 1:
                    partition = -1
            elif name in ROW_PARTITION_NAMES:
                # Fused SPM decoders launch these with partition=0. The shape
                # rule would say col here (both are legal at these dims), so
                # eager `aten::linear` cannot read the result — that is why
                # `_record_linear_layout` below installs the honouring forward.
                partition = 0
            else:
                partition = get_linear_partition(
                    in_features, out_features, dwidth,
                    num_cores=layer_num_cores,
                )

            if partition >= 0:
                with torch.no_grad():
                    if is_tied:
                        # Untie the weights: clone before transforming so embedding is not affected
                        child.weight = nn.Parameter(child.weight.data.clone())

                    # Transform weight
                    transformed = transform_linear_weight(child.weight.data, partition, num_cores=layer_num_cores)
                    child.weight.data = transformed

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
        else:
            # Recurse into child modules
            child_info = convert_linear_weights_inplace(child, full_name, _embedding_ptrs, force_col_partition, attn_num_cores, skip_names)
            partition_info.update(child_info)

    return partition_info


# Public names; compatibility aliases remain for existing converters.
swizzle_linear = transform_linear_weight
swizzle_model_inplace = convert_linear_weights_inplace


# Whole-model conversion entry point.
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
