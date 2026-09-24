# Performance measurement

[简体中文](performance.zh.md) | English

RhinoForge performance results are meaningful only for one exact model and
runtime profile. Use applicable correctness checks from
[Model validation policy](validation_policy.md).

Internal board measurements and performance comparisons stay outside the
repository, release notes and distribution artifacts. This guide describes how
to diagnose and tune an application; it does not publish benchmark results.

## Bind the measured profile

Record these fields before running:

- RhinoForge source revision and installed package version;
- Rhino Launch package and combined operator-asset compatibility identifiers;
- model checkpoint revision, precision, and quantization metadata;
- board/runtime version and relevant host software versions;
- input shape or request envelope, generation or action settings, and batch;
- TOML path and effective public runtime settings; and
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
3. Apply the same profile and complete graph preparation and 2 warmups. Use
   5–10 measured iterations unless a different protocol was requested.
4. Measure complete public API calls, including any output materialization
   required by the caller. Report sample count and median; report tail
   percentiles only with enough samples to interpret them.
5. Reuse a valid baseline. A small, interleaved baseline/candidate comparison
   is useful when environmental noise leaves a specific gain unresolved.
6. Run Torch profiling in a separate diagnostic process. Profiler-
   enabled latency does not represent normal inference.

For language models, distinguish prefill throughput, decode throughput, and
end-to-end request latency. Count logical input and generated tokens, not padded
execution rows. For VLM/VLA paths, report image/view count, prompt envelope,
denoise steps, action horizon, and end-to-end policy-call latency; use
`action chunks/s` only when the chunk definition is stated.

## Local measurement notes

Keep the following context with local measurements so they remain useful when
diagnosing a regression:

| Field | Required content |
|---|---|
| Profile | Model revision, precision/quantization, input envelope, and TOML path |
| Runtime | RhinoForge revision, Launch package, operator asset, and board/runtime version |
| Correctness | Checks performed, reference source, result, and uncovered scope |
| Timing | Boundary, warmup, sample count, summary statistics, and units |
| Lifecycle | Cold/startup or steady-state; Graph BUILD/REPLAY evidence where applicable |
| Conditions | Profiler off/on, host configuration relevant to the measurement, and date |

Torch traces can expose application shapes and source paths. Keep traces and
measurement reports outside the repository and release artifacts. See the
[profiling guide](model_testing.md#torch-profile) for the supported TOML entry
points.
