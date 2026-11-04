# Preview update — 2026-10-10

| Area | Changes |
|---|---|
| Qwen3.5 | Official 27B W8 Text/VL at 4/6/8 cores and W4 group32 at 8 cores; finalized weight upload, caching allocation and Graph resources |
| LingBot v2 | Runtime Graph entry, FP16/W8A16, two/three cameras, image/observation/case inputs and training configuration alignment |
| Graph | Per-instance `batch_snapshot_mode`, capability checks, automatic ordinary-replay fallback and public policy constructors |
| Memory | Bounded CPU/device synchronization, refreshed mutable inputs and reused persistent storage |
| Examples | Weight download/conversion guides and current configuration index |
| Runtime assets | Matching Launch, main and DDR-expansion assets |

Model and execution settings remain bound before weight installation. See [examples](../examples/README.md) and [runtime configuration](runtime_config.md) for accepted inputs and controls.
