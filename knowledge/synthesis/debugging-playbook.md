# Debugging and profiling playbook

Use this sequence for numerical mismatches, replay-only failures, unexpected
fallback, or performance regressions. It retains the reusable conclusions from
prior debugging practice without publishing session history or diagnostic
payloads.

## Fast triage order

1. **Bind the exact profile.** Record source revision, model/checkpoint revision,
   dtype, input envelope, TOML, Launch package, combined operator asset, and
   compatibility identifiers. Stop on any mismatch.
   [Release-quality checks](../../docs/model_testing.md#what-a-release-quality-model-check-still-needs)
2. **Reproduce both references and one RPU result.** Use the exact same-dtype
   executable for implementation parity and CPU FP32 as the precision anchor.
   Preserve output hashes and top-level numerical/task metrics before adding
   probes. Only if top-level evidence fails, compare the smallest divergent
   public model boundary.
   [Validation policy](../../docs/validation_policy.md)
3. **Prove the Graph lifecycle.** A repeated signature should show one BUILD,
   increasing REPLAY count, fixed cache size, no recapture, and a true cache
   invariant. Use `snapshot()`, `debug_bucket_counts()`,
   `dump_signature_tree()`, or `explain_miss()` to explain state; their output is
   diagnostic, not correctness evidence.
   [Graph API](../../docs/api_reference.md#graph-api)
4. **Repeat across warmup and input changes.** Check deterministic output across
   the supported warmup settings and exercise more than one real input. For
   image or multi-input paths, retain earlier outputs long enough to catch
   backing-storage reuse.
   [Porting pitfalls](../../docs/pitfalls.md)
5. **Check shared contracts before model math.** In order: weight conversion,
   replay-persistent SPM lifetime, address-versus-offset use, output ownership,
   DMA address lifetime, Graph signature completeness, model-state
   invalidation, then profile-specific computation.
   [Pitfalls overview](pitfalls-overview.md)
6. **Localize one boundary at a time.** Compare CPU and RPU at a stable public
   stage, identify the first transition, and split only that stage. A stage that
   receives different inputs may be an amplifier rather than the originating
   defect; use a shared-input A/B before changing product code.

## Multi-component policy checks

For Pi0.5, RhinoVLA, and other visual-language-action policies, keep the
preprocessed request and any stochastic starting state identical between the
reference and RPU paths. Check the visual output, prefix/text state, cache
progression, and action path independently before relying on the final action
chunk. A downstream component can consume the same wrong upstream state on two
comparison paths, so final-output agreement alone does not localize every
component.

For multiple images or repeated same-shape requests, retain the first output
while executing the next input and verify that its bytes and storage remain
independent. This catches output ownership mistakes that a one-input smoke test
cannot expose. [Porting pitfalls](../../docs/pitfalls.md)

## Performance diagnosis

Choose one mechanism according to the question:

| Question | Mechanism | Rule |
|---|---|---|
| Which backend wrappers or accumulated regions are expensive? | `torch.rpu.set_profile(True)` | Warm first when measuring steady state and reset counters before the window. |
| Where is host/operator time spent? | `torch.profiler.profile(...)` | Profile a bounded call; keep warmup separate and expect tracing overhead. |
| Which retained Graph segments consume device time? | `torch.rpu.hw_perf_trace(...)` | Enable before model construction and first BUILD; use a fresh process. |

See [runtime profiling controls](../../docs/runtime_config.md#three-profiling-mechanisms)
and the [TOML profiling guide](../../docs/model_testing.md#torch-profile).

Do not use profiler-on latency as the release number. Raw Torch and hardware
traces may contain tensor shapes, source paths, device addresses, and execution
structure. Review locally, publish only the necessary summary, and never add
generated trace directories to the repository.

## Stop conditions

- An end-to-end smoke run alone does not establish model support.
- A clean warning log does not establish Graph reuse.
- A Graph invariant does not establish numerical correctness.
- A faster run with changed output is a different numerical profile, not a
  performance result for the original profile.
- If one required reference, runtime asset, numerical gate, maximum-envelope
  run, or lifecycle check is missing, report it as pending or blocked.

## Sources

- [Model execution and profiling](../../docs/model_testing.md)
- [Runtime configuration](../../docs/runtime_config.md)
- [Graph API](../../docs/api_reference.md#graph-api)
- [Porting pitfalls](../../docs/pitfalls.md)
- [Model porting](../../docs/model_porting.md)
- [Model validation policy](../../docs/validation_policy.md)
- [GraphCache capture](../concepts/graphcache-capture.md)
