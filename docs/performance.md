# Performance measurement

RhinoForge performance results are meaningful only for one exact model and
runtime profile. Do not compare or publish a latency number until the same run
has passed the applicable gates in
[Model validation policy](validation_policy.md).

This repository intentionally does not carry a moving table of internal board
results. Release notes may publish measurements that are bound to an immutable
software, asset, model, input, and measurement record.

## Bind the measured profile

Record these fields before running:

- RhinoForge source revision and installed package version;
- Rhino Launch package and combined operator-asset compatibility identifiers;
- model checkpoint revision, file hashes, precision, and quantization metadata;
- board/runtime version and relevant host software versions;
- input shape or request envelope, generation or action settings, and batch;
- TOML hash and effective public runtime settings; and
- warmup count, measured sample count, and profiler state.

Run one model profile per fresh process. A different checkpoint, precision,
input envelope, graph plan, or runtime asset set is a different benchmark.
[Runtime profiles](../knowledge/concepts/runtime-profiles.md) explains why
individual environment switches cannot be compared in isolation.

## Separate startup and steady state

Report startup and steady-state latency separately:

- **Startup** may include package import, model loading, weight conversion,
  device transfer, Graph BUILD, and explicit graph preparation. State the exact
  boundary used.
- **Steady state** starts only after the declared warmup and Graph preparation.
  Reuse the same admitted input envelope and confirm that repeated signatures
  use the documented replay lifecycle.

Do not subtract selected host work from an end-to-end number. If a component is
reported separately, define its input/output boundaries and show that the sum
is not being presented as end-to-end latency.

## Measurement protocol

1. Fix deterministic or recorded inputs. For stochastic policies, reuse the
   same initial noise or random state for correctness and timing comparisons.
2. Run the required correctness and lifecycle checks without profilers.
3. Start a fresh process, apply the exact same profile, and complete the stated
   warmup or graph-preparation phase.
4. Measure complete public API calls, including any output materialization
   required by the caller. Report sample count plus at least median and a tail
   percentile; do not publish only the fastest run.
5. Repeat the baseline and candidate in interleaved or otherwise temperature-
   controlled order when evaluating a small optimization. Report absolute
   values as well as the delta.
6. Run Torch or hardware profiling in separate diagnostic processes. Profiler-
   enabled latency is not a release performance result.

For language models, distinguish prefill throughput, decode throughput, and
end-to-end request latency. Count logical input and generated tokens, not padded
execution rows. For VLM/VLA paths, report image/view count, prompt envelope,
denoise steps, action horizon, and end-to-end policy-call latency; use
`action chunks/s` only when the chunk definition is stated.

## Minimal publication record

Every published result should include:

| Field | Required content |
|---|---|
| Profile | Model revision, precision/quantization, input envelope, and TOML hash |
| Runtime | RhinoForge revision, Launch package, operator asset, and board/runtime version |
| Correctness | Gate name, reference identity, result, and any pending scope |
| Timing | Boundary, warmup, sample count, median, tail percentile, and units |
| Lifecycle | Cold/startup or steady-state; Graph BUILD/REPLAY evidence where applicable |
| Conditions | Profiler off/on, host configuration relevant to the measurement, and date |

Raw Torch and hardware traces can expose application shapes, source paths,
device addresses, and execution structure. Keep them out of the repository and
publish only the reviewed summary. See the
[profiling guide](model_testing.md#torch-profile) for the supported TOML entry
points.
