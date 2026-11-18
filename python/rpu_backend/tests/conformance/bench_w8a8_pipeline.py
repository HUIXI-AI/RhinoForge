#!/usr/bin/env python3
"""Bounded resident-input W8A16 versus device-quantized W8A8 A/B.

Both paths consume the same FP16 SPM input and packed INT8 W/FP16 SW. B includes
device activation quantization. Both row paths include the actual production
zero-residual all-reduce. Upload, tensor/SPM allocation, CPU references and output
reads are outside timing. Immediate mode includes host launcher overhead. Graph
mode builds A/B separately once, then uses retained replay with stable addresses
and skips the launcher stream. Both measure synchronous call latency, including
the shared Graph/SDK submission cost; neither is a model benchmark.
Each arm uses a contiguous warmup-then-measurement block (default W2/R5), with
Graph checks and output reads outside the block, to avoid predecessor-order bias.
--breakdown additionally times quantization alone and W8A8 with already generated
Q/SX, each with its own retained entry in Graph mode. Separate call latencies
include separate submission costs and must not be added as a same-Graph total.
--projections 3 uses the same input for three column projections, with independent
DDR weight/scale clones and SPM output slots. A executes three W8A16 projections;
B quantizes once and executes three W8A8 projections in one retained Graph. This
is a synthetic shared-input microbenchmark, not model QKV or task-quality evidence.
--norm layer_norm restricts this comparison to Vision QKV M512/M768 with three
projections: A runs production FP16 LayerNorm then W8A16; B fuses LayerNorm and
activation quantization before W8A8. Before timing, all cores' Q/SX bytes must
match separate production LayerNorm + quantization. Linear CPU references use
the actual baseline device LayerNorm output, not a substitute CPU LayerNorm.
--producer gelu_tanh measures GELU+FC2 with a nonzero compact residual consumer;
--producer owner_layer_norm measures the actual owner reduction/residual/LN2
and gather before FC1. Both are restricted to Vision M512/M768, TP8 and ACC16.
Their references use actual baseline device FP16 producer output; owner compact
raw and all-core Q/SX are checked byte-for-byte outside timing.
--tile-ordinal selects an installed W8A8 payload only in the five exact Qwen32
C128/C192 projections with TP8 and --execution graph.
The default -1 calls the unchanged production
launcher; explicit trials use the same ABI validation, typed owners and Graph
machinery from a test-only helper. They do not change model tile selection.

--list and --help require no torch, backend import, or device. Hardware execution
requires the existing board occupancy/locking protocol and a rebuilt conformance
library plus the complete current main/expansion REFs. No locks are acquired by
this driver. Default selection is one Pi C272 column and one row case.
The default A baseline is ACC16, matching the optimized Pi0.5 routes; ACC32
is selectable for separate diagnostics and is reported explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics

# name: (M, global N, global K, partition 1=col/0=row, bias).
# Vision dimensions are physical padded production dimensions.
CASES = {
    "pi_q_c272": (272, 2048, 2048, 1, False),
    "pi_down_c272": (272, 2048, 16384, 0, False),
    "pi_q_c400": (400, 2048, 2048, 1, False),
    "pi_down_c400": (400, 2048, 16384, 0, False),
    "vision_qkv_m512": (512, 1280, 1152, 1, True),
    "vision_o_m512": (512, 1152, 1280, 0, True),
    "vision_fc1_m512": (512, 4352, 1152, 1, True),
    "vision_fc2_m512": (512, 1152, 4352, 0, True),
    "vision_qkv_m768": (768, 1280, 1152, 1, True),
    "vision_o_m768": (768, 1152, 1280, 0, True),
    "vision_fc1_m768": (768, 4352, 1152, 1, True),
    "vision_fc2_m768": (768, 1152, 4352, 0, True),
    "qwen32_q_c128": (128, 8192, 5120, 1, False),
    "qwen32_kv_c128": (128, 1024, 5120, 1, False),
    "qwen32_gate_up_c128": (128, 25600, 5120, 1, False),
    "qwen32_o_c128": (128, 5120, 8192, 0, False),
    "qwen32_down_c128": (128, 5120, 25600, 0, False),
    "qwen32_gate_up_c192": (192, 25600, 5120, 1, False),
    "qwen32_down_c192": (192, 5120, 25600, 0, False),
    "qwen32_q_c192": (192, 8192, 5120, 1, False),
    "qwen32_kv_c192": (192, 1024, 5120, 1, False),
    "qwen32_o_c192": (192, 5120, 8192, 0, False),
}
DEFAULT_CASES = ("pi_q_c272", "pi_down_c272")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=DEFAULT_CASES)
    parser.add_argument(
        "--list", action="store_true", help="list shapes without device imports"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--baseline", choices=("acc16", "acc32"), default="acc16")
    parser.add_argument("--execution", choices=("immediate", "graph"), default="immediate")
    parser.add_argument(
        "--tile-ordinal", type=int, choices=range(-1, 19), default=-1,
        help="test-only installed general W8A8 tile for exact Qwen32 cases; -1 uses production selection",
    )
    parser.add_argument(
        "--norm", choices=("none", "layer_norm"), default="none",
        help="include LayerNorm producer for Vision QKV with --projections 3",
    )
    parser.add_argument(
        "--producer", choices=("none", "gelu_tanh", "owner_layer_norm"), default="none",
        help="include the real Vision FC2 GELU or FC1 owner-LN2 producer/consumer",
    )
    parser.add_argument(
        "--breakdown", action="store_true",
        help="also time quantization alone and W8A8 with resident Q/SX",
    )
    parser.add_argument(
        "--projections", type=int, choices=(1, 3), default=1,
        help="independent column projections sharing one input and quantization",
    )
    parser.add_argument(
        "--verify-eager", action="store_true",
        help="also check three untimed public eager calls against the SPM B result",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write raw timings and numerical diagnostics as JSON",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.warmup <= 10 or not 1 <= args.runs <= 30:
        parser.error("warmup must be 1..10 and runs must be 1..30")
    if len(args.cases) != len(set(args.cases)):
        parser.error("duplicate cases are not accepted")
    if args.tile_ordinal >= 0:
        if any(not name.startswith("qwen32_") for name in args.cases):
            parser.error("--tile-ordinal requires exact Qwen32 cases")
        if args.projections != 1 or args.norm != "none" or args.producer != "none" or args.verify_eager:
            parser.error("--tile-ordinal requires one ordinary projection and no --verify-eager")
        if args.execution != "graph":
            parser.error("--tile-ordinal requires --execution graph; preparation is outside timing")
    if not args.list and args.producer != "none":
        family = "fc2" if args.producer == "gelu_tanh" else "fc1"
        if any(name not in (f"vision_{family}_m512", f"vision_{family}_m768") for name in args.cases):
            parser.error(f"--producer {args.producer} requires Vision {family.upper()} M512/M768")
        if args.norm != "none" or args.projections != 1 or args.baseline != "acc16" or args.verify_eager:
            parser.error("--producer requires --norm none, --projections 1, --baseline acc16 and no --verify-eager")
    if not args.list and args.norm == "layer_norm":
        if args.projections != 3 or any(
            name not in ("vision_qkv_m512", "vision_qkv_m768") for name in args.cases
        ):
            parser.error("--norm layer_norm requires Vision QKV M512/M768 and --projections 3")
        if args.verify_eager:
            parser.error("--verify-eager does not support --norm layer_norm")
    if not args.list and args.projections == 3:
        if any(CASES[name][3] != 1 for name in args.cases):
            parser.error("--projections 3 only supports column cases")
        if args.verify_eager:
            parser.error("--verify-eager only supports --projections 1")
    return args


def _valu_round(value):
    """R1 scale-product rounding: FP32 exponent, ten fraction bits."""
    import torch

    bits = value.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded = (bits + 4095 + ((bits >> 13) & 1)) & 0xFFFFE000
    return rounded.to(torch.int32).view(torch.float32)


def _ring_reference(partials, residual=None):
    """Use rounded FP16 partials and the production rotated source order."""
    import torch

    cores, shape = len(partials), partials[0].shape
    flat = [value.half().flatten().float() for value in partials]
    total = flat[0].numel()
    chunk = (((total + cores - 1) // cores + 15) // 16) * 16
    result = torch.zeros(total, dtype=torch.float32)
    for owner in range(cores):
        start, end = min(owner * chunk, total), min((owner + 1) * chunk, total)
        acc = torch.zeros(end - start, dtype=torch.float32)
        for step in range(cores):
            acc += flat[(owner + step) % cores][start:end]
        if residual is not None:
            acc += residual.flatten().float()[start:end]
        result[start:end] = acc.half().float()
    return result.reshape(shape)


def semantic_references(x, qw, sw, bias, partition, cores=8, residual=None):
    """Per-family mathematical references; numerical differences are diagnostics.

    A's diagnostic decodes scaled weights to FP16 before a FP32 GEMM; it does
    not emulate ACC16 accumulation. B quantizes each
    physical local activation shard, accumulates INT8 products, and applies SX
    then SW with R1 VALU rounding. Both round partial Y/bias to FP16 before row
    reduction. CPU GEMM accumulation order need not bitmatch the RPU VMAT.
    """
    import torch

    n, k = qw.shape
    local_n = n // cores if partition else n
    local_k = k if partition else k // cores
    a, b = [], []
    for core in range(cores):
        xs = x if partition else x[:, core * local_k : (core + 1) * local_k]
        ws = (
            qw[core * local_n : (core + 1) * local_n]
            if partition
            else qw[:, core * local_k : (core + 1) * local_k]
        )
        scale = sw[core * local_n : (core + 1) * local_n] if partition else sw
        decoded_w = (ws.float() * scale.float()[:, None]).half().float()
        ya = (xs.float() @ decoded_w.T).half()
        sx = (xs.float().abs().amax(dim=1) / 127).clamp_min(1e-6).half()
        qx = (xs.float() / sx.float()[:, None]).half().round().clamp(-127, 127)
        raw = qx.float() @ ws.float().T
        yb = _valu_round(
            _valu_round(_valu_round(raw) * sx.float()[:, None]) * scale.float()
        ).half()
        if bias is not None and (partition or core == 0):
            bc = bias[core * local_n : (core + 1) * local_n] if partition else bias
            ya = (ya.float() + bc.float()).half()
            yb = (yb.float() + bc.float()).half()
        a.append(ya)
        b.append(yb)
    if partition:
        return torch.cat(a, dim=1).float(), torch.cat(b, dim=1).float()
    return _ring_reference(a, residual), _ring_reference(b, residual)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for name, (m, n, k, partition, bias) in CASES.items():
            print(
                f"{name}: M={m} N={n} K={k} {'col' if partition else 'row'} TP8 bias={bias}"
            )
        return

    # All device-related imports are deliberately after argument-only modes.
    import torch
    import rpu_backend  # noqa: F401
    from rpu_backend.runtime.weights import (
        tp_col_swizzle_mc_weight,
        tp_row_swizzle_mc_weight,
    )
    from rpu_backend.tests.conformance._numeric import numeric_metrics

    library = Path(
        os.environ.get(
            "RPU_CONFORMANCE_SO", Path(__file__).parent / "_cpp/librpu_conformance.so"
        )
    ).resolve()
    if not library.is_file():
        raise SystemExit(
            f"Missing conformance library: {library}; rebuild with BUILD_CONFORMANCE=ON"
        )
    torch.ops.load_library(str(library))
    torch.manual_seed(42)
    rows = []
    report = {
        "scope": f"resident-input synchronous {args.execution} W8A16 {args.baseline.upper()} vs quantize+W8A8; row includes identical zero-residual ring",
        "baseline": args.baseline,
        "execution": args.execution,
        "breakdown": args.breakdown,
        "projections": args.projections,
        "tile_ordinal": args.tile_ordinal,
        "tile_selection": "production" if args.tile_ordinal < 0 else "explicit installed payload; test-only",
        "qwen_projection_scope": "individual physical projection; the model shares QKV/Gate-Up quantization" if any(name.startswith("qwen32_") for name in args.cases) else None,
        "projection_scope": "same-input independent W/SW clones and output slots; synthetic microbenchmark, not model QKV or task-quality evidence" if args.projections == 3 else "single projection",
        "projection_execution": {"a_pl_calls": args.projections, "b_quantize_calls": 1, "b_pl_calls": args.projections},
        "untimed_builds_per_side": 1 if args.execution == "graph" else 0,
        "warmup_kind": "contiguous_per_arm_retained_replay" if args.execution == "graph" else "contiguous_per_arm_immediate_launch",
        "graph_replay": "stable-address skip_op_stream" if args.execution == "graph" else None,
        "timing_unit": "ns",
        "timing_columns": [f"w8a16_{args.baseline}", "quantize_w8a8"]
        + (["quantize_only", "w8a8_only"] if args.breakdown else []),
        "warmup_per_side": args.warmup,
        "measured_visits_per_side": args.runs,
        "order": "per-arm contiguous warmup then measured block: A,B,Q,P" if args.breakdown else "per-arm contiguous warmup then measured block: A,B",
        "untimed_output_validation_launches_per_side": 1,
        "untimed_breakdown_validation": "quantize_only then w8a8_only, output exactly equals B" if args.breakdown else None,
        "untimed_immediate_quantize_preparation": 1 if args.breakdown and args.execution == "immediate" else 0,
        "breakdown_timing_note": "independent synchronous calls include separate submission overhead; do not sum as same-Graph latency" if args.breakdown else None,
        "numerical_policy": "shape/dtype/finite required; error distributions record-only; no model quality claim",
        "a_reference": "FP16 decoded weights, FP32 mathematical GEMM; not exact ACC16 accumulation",
        "untimed_eager_calls_per_case": 3 if args.verify_eager else 0,
        "conformance_library": str(library),
        "rows": rows,
    }
    norm_enabled = args.norm == "layer_norm"
    producer_enabled = args.producer != "none"
    owner_enabled = args.producer == "owner_layer_norm"
    gelu_enabled = args.producer == "gelu_tanh"
    norm_eps = 1e-6
    if norm_enabled:
        report.update({
            "norm": "layer_norm",
            "norm_eps": norm_eps,
            "scope": f"resident raw-input synchronous {args.execution}: LayerNorm+3xW8A16 {args.baseline.upper()} vs fused LayerNorm-quantize+3xW8A8",
            "projection_execution": {"a_layer_norm_calls": 1, "a_pl_calls": 3,
                                     "b_layer_norm_quantize_calls": 1, "b_pl_calls": 3},
            "timing_columns": [f"layer_norm_w8a16_{args.baseline}", "layer_norm_quantize_w8a8"]
                + (["layer_norm_quantize_only", "w8a8_only"] if args.breakdown else []),
            "a_reference": "actual baseline device FP16 LayerNorm output, FP16 decoded weights, FP32 mathematical GEMM; not exact ACC16 accumulation",
            "b_reference": "actual baseline device FP16 LayerNorm output then INT8 quantization and R1 scale arithmetic",
            "untimed_norm_quantization_parity": "separate production LayerNorm + resident quantization vs fused LayerNorm-quantization; all-core Q/SX bytes including allocated padding and guards",
            "untimed_breakdown_validation": "layer_norm_quantize_only then w8a8_only, output exactly equals B" if args.breakdown else None,
            "breakdown_timing_note": "Q includes LayerNorm and quantization; A includes LayerNorm; separate synchronous call times are not additive" if args.breakdown else None,
        })
    if producer_enabled:
        producer_label = "owner_ln" if owner_enabled else "gelu"
        consumer_label = " plus compact residual consumer" if gelu_enabled else ""
        report.update({
            "producer": args.producer,
            "scope": f"resident-input synchronous {args.execution}: real {args.producer}+W8A16 ACC16 vs fused producer-quantize+W8A8{consumer_label}",
            "projection_execution": {"a_producer_calls": 1, "a_pl_calls": 1,
                                     "b_fused_producer_quantize_calls": 1, "b_pl_calls": 1,
                                     "compact_residual_calls_per_pl_arm": int(gelu_enabled)},
            "timing_columns": [f"{producer_label}_w8a16", f"{producer_label}_quantize_w8a8"]
                + ([f"{producer_label}_quantize_only", "w8a8_compact_residual_only" if gelu_enabled else "w8a8_only"] if args.breakdown else []),
            "a_reference": "actual baseline device FP16 producer output, FP16 decoded weights, FP32 mathematical GEMM; not exact ACC16 accumulation",
            "b_reference": "actual baseline device FP16 producer output then INT8 quantization and R1 scale arithmetic",
            "residual": "nonzero FP16; compact owner rows for FC2, replicated full residual for owner LN2",
            "untimed_producer_quantization_parity": "all-core Q/SX byte equality against actual FP16 producer then resident quantization; guards checked",
            "untimed_compact_raw_parity": "old owner compact raw exact against fused owner before timing and after replays" if owner_enabled else None,
            "untimed_output_all_core_gather": True if gelu_enabled else None,
            "breakdown_timing_note": "Q includes the fused producer; P includes compact residual for FC2; separate synchronous call times are not additive" if args.breakdown else None,
        })
    for name in args.cases:
        m, n, k, partition, has_bias = CASES[name]
        def phase(label):
            print(f"W8A8_PHASE case={name} execution={args.execution} phase={label}", flush=True)

        phase(f"case_begin M={m} N={n} K={k} projections={args.projections}")
        x = (torch.rand((8, m, k) if owner_enabled else (m, k)) * 0.2 - 0.1).half()
        if not owner_enabled:
            x[0] = 0
        original_w = (torch.rand(n, k) * 0.2 - 0.1).half()
        sw = (original_w.float().abs().amax(dim=1) / 127).clamp_min(1e-6).half()
        qw = (
            (original_w.float() / sw.float()[:, None])
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
        )
        bias = (torch.rand(n) * 0.2 - 0.1).half() if has_bias else None
        gamma = (torch.rand(k) * 0.2 + 0.9).half() if norm_enabled or owner_enabled else None
        beta = (torch.rand(k) * 0.2 - 0.1).half() if norm_enabled or owner_enabled else None
        residual = (torch.rand(m, 1152) * 0.2 - 0.1).half() if producer_enabled else None
        pack = tp_col_swizzle_mc_weight if partition else tp_row_swizzle_mc_weight
        packed = pack(qw, 8, dwidth=1)
        # Weight preparation/upload and references stay outside the native timer.
        if not norm_enabled and not producer_enabled:
            phase("reference_begin")
            expected_a, expected_b = semantic_references(x, qw, sw, bias, partition)
            if args.projections == 3:
                expected_a = expected_a.unsqueeze(0).expand(3, -1, -1)
                expected_b = expected_b.unsqueeze(0).expand(3, -1, -1)
            phase("reference_done")
        phase("upload_x_begin")
        xd = x.to("rpu")
        phase("upload_weight_begin")
        wd = packed.to("rpu")
        phase("upload_scale_bias_begin")
        sd = sw.to("rpu")
        bd = None if bias is None else bias.to("rpu")
        gd = gamma.to("rpu") if gamma is not None else None
        betad = beta.to("rpu") if beta is not None else None
        rd = residual.to("rpu") if residual is not None else None
        phase("upload_done")
        phase("native_begin")
        if producer_enabled:
            out_a, out_b, times, baseline_norm = torch.ops.rpu.test_w8a8_producer_pipeline_bench(
                xd, wd, sd, bd, rd, gd, betad, args.producer, norm_eps,
                args.warmup, args.runs, args.execution == "graph", args.breakdown,
            )
        elif norm_enabled:
            out_a, out_b, times, baseline_norm = torch.ops.rpu.test_w8a8_norm_pipeline_bench(
                xd, wd, sd, bd, gd, betad, norm_eps, 8, args.warmup, args.runs,
                args.baseline == "acc32", args.execution == "graph", args.breakdown,
            )
        else:
            out_a, out_b, times = torch.ops.rpu.test_w8a8_pipeline_bench(
                xd, wd, sd, bd,
                partition,
                8,
                args.warmup,
                args.runs,
                args.baseline == "acc32",
                args.execution == "graph",
                args.breakdown,
                args.projections,
                args.tile_ordinal,
            )
        phase("native_done")
        norm_metrics = {}
        if producer_enabled:
            if (baseline_norm.device.type != "cpu" or baseline_norm.dtype != torch.float16
                    or baseline_norm.shape != (m, k) or not torch.isfinite(baseline_norm).all()):
                raise AssertionError("invalid baseline device producer reference output")
            phase("reference_from_device_producer_begin")
            expected_a, expected_b = semantic_references(
                baseline_norm, qw, sw, bias, partition,
                residual=residual if gelu_enabled else None,
            )
            norm_metrics = {
                "producer_quantize_all_core_q_sx_byte_parity": True,
                "producer_quantize_spm_guards_intact": True,
                "producer_reference_source": "baseline production device owner-LN2 FP16 gathered output" if owner_enabled else "baseline production device GELU FP16 output, gathered from all eight TP shards",
                "owner_compact_raw_exact_before_and_after_replay": True if owner_enabled else None,
                "compact_residual_output_all_core_equal": True if gelu_enabled else None,
            }
            if gelu_enabled:
                norm_metrics["baseline_device_gelu_vs_fp32_torch_diagnostic"] = numeric_metrics(
                    baseline_norm.float(), torch.nn.functional.gelu(x.float(), approximate="tanh")
                )
            phase("reference_from_device_producer_done")
        elif norm_enabled:
            if (baseline_norm.device.type != "cpu" or baseline_norm.dtype != torch.float16
                    or baseline_norm.shape != (m, k) or not torch.isfinite(baseline_norm).all()):
                raise AssertionError("invalid baseline device LayerNorm reference output")
            phase("reference_from_device_norm_begin")
            expected_a, expected_b = semantic_references(baseline_norm, qw, sw, bias, partition)
            expected_a = expected_a.unsqueeze(0).expand(3, -1, -1)
            expected_b = expected_b.unsqueeze(0).expand(3, -1, -1)
            fp32_norm = torch.nn.functional.layer_norm(
                x.float(), (k,), gamma.float(), beta.float(), norm_eps
            )
            norm_metrics = {
                "baseline_device_norm_vs_fp32_torch_diagnostic": numeric_metrics(baseline_norm.float(), fp32_norm),
                "norm_quantize_all_core_q_sx_byte_parity": True,
                "norm_quantize_spm_guards_intact": True,
                "norm_reference_source": "baseline production device LayerNorm FP16 output from core0",
            }
            phase("reference_from_device_norm_done")
        metrics_a = numeric_metrics(out_a, expected_a)
        metrics_b = numeric_metrics(out_b, expected_b)
        metrics_quantization = numeric_metrics(expected_b, expected_a)
        projection_metrics = {}
        if args.projections == 3:
            projection_metrics["per_projection_metrics"] = [
                {
                    "projection": projection,
                    "a_vs_fp32_math_metrics": numeric_metrics(out_a[projection], expected_a[projection]),
                    "b_same_semantic_metrics": numeric_metrics(out_b[projection], expected_b[projection]),
                }
                for projection in range(args.projections)
            ]
        eager_metrics = []
        if args.verify_eager:
            outputs = []
            for invocation in range(3):
                phase(f"eager_begin call={invocation + 1}")
                outputs.append(torch.ops.rpu.linear_w8a8(xd, wd, sd, bd, partition, 8))
                phase(f"eager_done call={invocation + 1}")
            if len({value.data_ptr() for value in outputs}) != len(outputs):
                raise AssertionError("eager calls must return independent output owners")
            for invocation, value in enumerate(outputs):
                phase(f"eager_readback_begin call={invocation + 1}")
                actual = value.cpu().float()
                eager_metrics.append(numeric_metrics(actual, expected_b))
                if not torch.equal(actual, out_b):
                    raise AssertionError("public eager and SPM W8A8 paths differ for the same input")
                phase(f"eager_readback_done call={invocation + 1}")
        if (
            times.shape != (args.runs, 4 if args.breakdown else 2)
            or times.dtype != torch.float64
            or not torch.isfinite(times).all()
            or not (times > 0).all()
        ):
            raise AssertionError("invalid timing table")
        raw = times.tolist()
        median_a = statistics.median(row[0] for row in raw)
        median_b = statistics.median(row[1] for row in raw)
        breakdown_metrics = {}
        if args.breakdown:
            median_quant = statistics.median(row[2] for row in raw)
            median_pl = statistics.median(row[3] for row in raw)
            breakdown_metrics = {
                "median_quantize_only_ns": median_quant,
                "median_w8a8_only_ns": median_pl,
                "w8a8_only_vs_a_latency_change_percent": 100 * (median_pl / median_a - 1),
                "standalone_quantize_pl_equals_b": True,
            }
            if norm_enabled:
                breakdown_metrics["quantize_only_scope"] = "fused LayerNorm and quantization"
                breakdown_metrics["w8a8_only_comparison_baseline"] = "LayerNorm plus three W8A16 projections"
            if producer_enabled:
                breakdown_metrics["quantize_only_scope"] = args.producer + " fused with quantization"
                breakdown_metrics["w8a8_only_scope"] = "W8A8 plus compact residual" if gelu_enabled else "W8A8 FC1 only"
                breakdown_metrics["w8a8_only_comparison_baseline"] = args.producer + " plus W8A16" + (" plus compact residual" if gelu_enabled else "")
        rows.append(
            {
                "name": name,
                "shape_m_n_k": [m, n, k],
                "partition": "col" if partition else "row",
                "bias": has_bias,
                "projections": args.projections,
                "tile_ordinal": args.tile_ordinal,
                "samples_ns": raw,
                "median_a_ns": median_a,
                "median_b_ns": median_b,
                "latency_change_percent": 100 * (median_b / median_a - 1),
                "a_vs_fp32_math_metrics": metrics_a,
                "b_same_semantic_metrics": metrics_b,
                "quantized_reference_vs_a16_reference_metrics": metrics_quantization,
                "eager_same_semantic_metrics": eager_metrics,
                "eager_equals_spm": True if args.verify_eager else None,
                **breakdown_metrics,
                **projection_metrics,
                **norm_metrics,
            }
        )
        print(
            f"{name}: projections={args.projections} A={median_a / 1000:.3f} us B={median_b / 1000:.3f} us change={100 * (median_b / median_a - 1):+.2f}%"
        )
        if args.breakdown:
            print(
                f"{name}: {'norm_quant_only' if norm_enabled else 'quant_only'}={median_quant / 1000:.3f} us "
                f"PL_only={median_pl / 1000:.3f} us "
                f"PL_vs_A={100 * (median_pl / median_a - 1):+.2f}% "
                "(separate synchronous calls; not additive)"
            )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        phase("case_done")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
