# Model execution and profiling

[简体中文](model_testing.zh.md) | English

The model scripts in `examples/` are direct entry points. The unified runner
applies documented TOML environment settings and optional Torch profiling to
the same scripts. Configurations are archived by model; see
[model examples](../examples/README.md) for paths and precision semantics.
The [full catalog](../examples/configs/README.md) includes Qwen3.5 Text and VL,
separate Qwen3-VL components, and precision, core-count and input-length variants.
Its `[example]/[input]/[run]` format (Pi0.5 uses `[pi05]`) works through the same
entry points. A few independent integration templates use `[runner]/[model]`;
each model has one canonical layout, without duplicate compatibility templates.
Do not mix fields from the two formats.

## Run one configuration

```sh
python examples/run_model.py --config examples/configs/qwen3/text/0_6b/fp16.toml --check-config
python examples/run_model.py --config examples/configs/qwen3_vl/vl/4b/w8a16.toml
python examples/pi05.py --config examples/configs/pi05/2cam/w8a16_action_nvfp4.toml
```

`--check-config` does not load a model or access a board. Replace placeholder
paths before execution. Model, image, batch and output paths in TOML are
relative to the command's working directory. Quantized directories must carry
the metadata expected by their loader; changing a filename does not convert
weights. See [quantization](quantization.md).

Examples requiring preprocessed tensors use caller-supplied CPU tensor mappings
from the selected public checkpoint's processor. A component example does not
represent the complete robot policy or application. Runtime checks govern the
admitted input envelope. Historical qualification records do not define the
current implementation or expand the scope of an example.

## Torch profile

Qwen and policy templates use `[run].profile = true` or `--profile`. Independent
integration templates using `[runner/model/request]` use this section:

```toml
[runner.torch_profile]
enabled = true
output = "profiles/torch_trace.json"
record_shapes = false
profile_memory = false
with_stack = false
```

For catalog configurations, use `[run].profile = true` or `--profile` instead.
This collects one extra diagnostic inference after the timed runs. Set
`run.warmup` and `run.runs` explicitly; short comparisons can use one warmup and
two or three samples with `input.decode_steps = 32`. Preserve all other input
and execution settings, and keep `warmup_decode_steps` at or below that limit.
Decode steps count calls after the first prefill token; EOS can stop them early.
Long-decode templates run only when explicitly selected. Example images are
placeholders, not the fixtures used for earlier performance reports.

The profiler wraps one execution. Profile collection changes timing, so collect
unprofiled latency separately. Keep shape, memory and stack collection disabled
unless the diagnosis needs them. The trace output can contain application data.
For the complete runner settings see [runtime configuration](runtime_config.md).

## Hardware performance trace

The public example runner rejects `hwperf=true` during configuration checks.
Use the installed runtime's hardware profiler only for a specific unresolved
operator or synchronization question. Keep tracing separate from steady-state
latency measurement. Record the exact kernel asset and input shape, and retain
low-level traces with the application owner rather than in the source tree.

## Focused checks

Check that the board is idle, acquire its device lock and use an isolated
environment bound to the intended worktree. Use one representative case per
changed path: normally 3–10 replays, or 2 warmups and 5–10 performance samples.
Confirm computational-path correctness before recording accumulated numerical
differences. Stop after the affected checks pass; see
[validation](validation_policy.md) and [performance](performance.md).
