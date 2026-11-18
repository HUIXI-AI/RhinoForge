#!/usr/bin/env python3
"""Graph ``end()`` lets other Python threads run while the device executes (board test).

A CPU worker thread (the G0.5 LIBERO ActionCodec overlap) only makes progress during
a Graph replay when ``end()`` releases the GIL; holding it starved the worker for
the whole device wait.

Run:
    sudo LD_PRELOAD=$(g++ -print-file-name=libstdc++.so.6) $RPU_PYTHON -u \\
        python/rpu_backend/tests/graph/test_graph_end_releases_gil.py
"""
from __future__ import annotations

import threading
import time

import torch

import rpu_backend


def _signature():
    return rpu_backend.graph.GraphSignature(
        op_id="graph_end_releases_gil", shapes=[1, 1024], dyn_dims=[4096],
        dtypes=[torch.float16])


def _chain(x: torch.Tensor) -> torch.Tensor:
    for _ in range(8):
        x = torch.nn.functional.gelu(x)
    return x


def test_replay_end_lets_another_python_thread_run() -> None:
    torch.rpu.set_caching_allocator(True)
    cache = rpu_backend.graph.GraphCache()
    x = torch.randn(1, 1024, 4096, dtype=torch.float16).to("rpu")
    with cache.capture(_signature()) as graph:
        built = _chain(x)
    assert graph.state() == 2, "the Graph did not BUILD"
    want = built.cpu()
    ticks = []
    stop = threading.Event()

    def tick():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.0002)
    worker = threading.Thread(target=tick, daemon=True)
    worker.start()
    windows = []
    try:
        for _ in range(3):
            with cache.capture(_signature()) as graph:
                assert graph.state() == 3, "expected REPLAYING"
                out = _chain(x)
                start = time.perf_counter()
            windows.append((start, time.perf_counter()))
    finally:
        stop.set()
        worker.join()
    assert torch.equal(out.cpu().view(torch.int16), want.view(torch.int16))
    for start, end in windows:
        inside = [t for t in ticks if start < t < end]
        span_ms = (end - start) * 1e3
        print(f"end() {span_ms:.2f} ms, worker ticks inside {len(inside)}", flush=True)
        assert span_ms > 2.0, "replay too short to show the GIL; enlarge the chain"
        assert len(inside) >= 3, f"worker starved for {span_ms:.2f} ms inside end()"


if __name__ == "__main__":
    test_replay_end_lets_another_python_thread_run()
    print("[PASS] Graph end() releases the GIL")
