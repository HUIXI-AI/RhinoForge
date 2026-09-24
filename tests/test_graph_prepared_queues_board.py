"""Explicit board probe for cross-entry, multi-segment prepared Graph replay.

Run only under the device locks held by the integration coordinator:
  python tests/test_graph_prepared_queues_board.py --run-board --output result.json

Default pytest collection does not invoke hardware. This needs no model weights.
--require-prepared-hits is for a separately built candidate with bounded segment
retention enabled after reviewing the old LingBot2 stale-buffer failure.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time


def run_board(rounds: int, require_hits: bool) -> dict:
    import torch
    import rpu_backend
    from rpu_backend.graph import GraphCache, GraphRuntimePolicy, GraphSignature

    if rounds < 4:
        raise ValueError("at least four rounds are required")
    # Lower only this probe's graph segmentation budget. The actual SDK load
    # and global resource settings stay untouched; the smaller bound forces a
    # tiny graph to exercise multiple prepared batches without thousands of ops.
    policy = replace(GraphRuntimePolicy.from_environment(),
                     lkn_max_batch_entries=8, fast_replay_skip_sync=True)
    caches = [GraphCache(max_entries=1, runtime_policy=policy) for _ in range(2)]
    shape = (32, 64)
    inputs = [[torch.empty(shape, dtype=torch.uint8, device="rpu") for _ in range(3)] for _ in caches]
    outputs = [[torch.empty(shape, dtype=torch.float16, device="rpu") for _ in range(3)] for _ in caches]
    signatures = [GraphSignature(op_id=f"prepared_queue_probe_{i}", shapes=shape,
                                 dtypes=[torch.float16]) for i in range(2)]
    timings = [[], []]
    after_warmup = []
    expected_by_owner = [None, None]
    retained = None
    retained_copy = None
    retained_export = None

    def require_equal(actual_cpu, expected_cpu, *, check, iteration,
                      executing_owner, observed_owner, op, source=None):
        if torch.equal(actual_cpu, expected_cpu):
            return
        actual_flat = actual_cpu.reshape(-1)
        expected_flat = expected_cpu.reshape(-1)
        mismatches = (actual_flat != expected_flat).nonzero().flatten()
        indices = mismatches[:8].tolist()
        raise AssertionError(json.dumps({
            "check": check, "iteration": iteration,
            "executing_owner": executing_owner, "observed_owner": observed_owner,
            "op": op, "shape": list(actual_cpu.shape),
            "source_data_ptr": None if source is None else source.data_ptr(),
            "actual_cpu_data_ptr": actual_cpu.data_ptr(),
            "expected_cpu_data_ptr": expected_cpu.data_ptr(),
            "cpu_aliases_source": None if source is None else actual_cpu.data_ptr() == source.data_ptr(),
            "mismatch_count": mismatches.numel(), "first_indices": indices,
            "actual_values": actual_flat[indices].tolist(),
            "expected_values": expected_flat[indices].tolist(),
            "retained_export": retained_export,
        }, sort_keys=True))

    try:
        for iteration in range(rounds + 2):
            for owner, cache in enumerate(caches):
                # Every owner, operation and replay gets distinguishable bytes.
                cpu_inputs = [((torch.arange(2048).reshape(shape) + iteration*17 + owner*71 + op*23) % 256).to(torch.uint8)
                              for op in range(3)]
                for target, value in zip(inputs[owner], cpu_inputs):
                    target.copy_(value)
                before = cache.snapshot()
                start = time.perf_counter_ns()
                with cache.capture(signatures[owner]):
                    for source, target in zip(inputs[owner], outputs[owner]):
                        torch.ops.rpu.cast_uint8_fp16_into(source, target)
                duration_us = (time.perf_counter_ns() - start) / 1000
                if iteration >= 2:
                    timings[owner].append(duration_us)
                after = cache.snapshot()
                assert len(after) == cache.size() == 1 and cache.cache_invariant_ok()
                entry = after[0]
                assert entry.segment_count > 1, "probe must execute a multi-segment Graph"
                assert entry.kernel_count == 3 and entry.recapture_count == 0
                assert not entry.non_replayable_reason
                if before:
                    assert entry.graph_lifetime_id == before[0].graph_lifetime_id
                    assert entry.replay_count == before[0].replay_count + 1
                expected_by_owner[owner] = [value.half() for value in cpu_inputs]
                # Validate the inactive owner's buffers too: current-output
                # equality alone could miss a write into another retained entry.
                for observed_owner, expected_outputs in enumerate(expected_by_owner):
                    if expected_outputs is None:
                        continue
                    for op, (actual, expected) in enumerate(zip(outputs[observed_owner], expected_outputs)):
                        require_equal(actual.cpu(), expected,
                                      check="active_output" if observed_owner == owner else "inactive_owner_output",
                                      iteration=iteration, executing_owner=owner,
                                      observed_owner=observed_owner, op=op, source=actual)
                if retained is not None:
                    require_equal(retained, retained_copy, check="independent_cpu_snapshot",
                                  iteration=iteration, executing_owner=owner,
                                  observed_owner=0, op=0)
                else:
                    # This backend intentionally exports a shared-DDR CPU view
                    # for a contiguous same-dtype .cpu(). The caller-owned out
                    # tensor is reused every iteration, so clone explicitly when
                    # retaining an independent result across those writes.
                    cpu_view = outputs[owner][0].cpu()
                    retained = cpu_view.clone()
                    retained_copy = retained.clone()
                    retained_export = {
                        "owner": owner, "iteration": iteration, "op": 0,
                        "source_data_ptr": outputs[owner][0].data_ptr(),
                        "cpu_view_data_ptr": cpu_view.data_ptr(),
                        "cpu_aliases_source": cpu_view.data_ptr() == outputs[owner][0].data_ptr(),
                        "snapshot_data_ptr": retained.data_ptr(),
                        "reference_data_ptr": retained_copy.data_ptr(),
                    }
                if iteration == 1:
                    cache.freeze()
                    after_warmup.append(entry)
        result = []
        for owner, cache in enumerate(caches):
            entry = cache.snapshot()[0]
            warm = after_warmup[owner]
            hits = entry.prepared_segment_hit_total - warm.prepared_segment_hit_total
            misses = entry.prepared_segment_miss_total - warm.prepared_segment_miss_total
            if require_hits:
                assert hits == rounds * entry.segment_count and misses == 0, (hits, misses, entry.segment_count)
            ordered = sorted(timings[owner])
            result.append({"owner": owner, "segments": entry.segment_count,
                           "replays": entry.replay_count, "prepared_hits": hits,
                           "prepared_misses": misses,
                           "capture_p50_us": statistics.median(ordered),
                           "capture_p95_us": ordered[min(len(ordered)-1, int(len(ordered)*0.95))]})
        return {"source": str(Path(rpu_backend.__file__).resolve()),
                "kind": "unprofiled cast graph host-capture probe, no model performance claim",
                "changed_input_exact": True, "inactive_owner_outputs_unchanged": True,
                "independent_cpu_snapshot_unchanged": True,
                "retained_export": retained_export, "owners": result}
    finally:
        for cache in caches:
            cache.begin_warmup()
            cache.clear()
            assert cache.size() == 0 and cache.cache_invariant_ok()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-board", action="store_true")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--require-prepared-hits", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_board:
        parser.error("board execution requires explicit --run-board and the coordinator's device locks")
    try:
        result = run_board(args.rounds, args.require_prepared_hits)
    except BaseException as error:
        args.output.write_text(json.dumps({"status": "failed", "error_type": type(error).__name__,
                                          "error": str(error)}, indent=2) + "\n")
        raise
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
