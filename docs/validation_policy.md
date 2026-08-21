# Model validation policy

RhinoForge validates an exact model profile, not a family name. A profile binds
the checkpoint revision, preprocessing, precision, input envelope, execution
configuration, Rhino Launch package, combined operator asset, and RhinoForge
commit. Results do not transfer to another size, quantization format, or input
envelope without an explicit review.

Floating-point results can differ across devices and implementations even when
the mathematical computation is equivalent. A single global cosine threshold,
especially the minimum over every token or feature row, is therefore not a
sufficient product-quality decision. RhinoForge uses independent semantic,
implementation-parity, task-quality, and runtime-lifecycle gates.

## 1. Freeze the evidence identity

Record these fields before evaluation:

- source, dependency, Launch package, and operator-asset versions;
- checkpoint, tokenizer or preprocessor, and configuration hashes;
- precision and quantization recipe;
- input dataset or fixture revision, random seed, and sampled noise where
  applicable; and
- admitted sequence, image, batch, cache, and execution settings.

A CPU FP32 golden can be reused while every reference-side identity field and
input hash remains unchanged. A backend-only source change requires a new RPU
comparison, not a new CPU golden.

## 2. Hard semantic and runtime gates

These checks are binary and cannot be waived by a good end-to-end score:

- output shapes, dtypes, ordering, masks, positions, cache updates, and logical
  lengths match the model contract;
- outputs and states that the public contract requires to be finite contain no
  unexpected NaN or infinity (explicit mask sentinels remain governed by their
  operator contract), and caller-retained outputs use independent storage;
- unsupported profiles fail before irreversible weight transformation;
- exactly-once weight transformation, Graph signatures, DMA ownership, and
  persistent-state lifetimes are correct;
- repeated runs are stable under the profile's declared repeatability contract;
  stochastic profiles compare the same frozen seed, noise, or sampled state;
  and
- the applicable Graph path proves BUILD followed by stable REPLAY, or proves
  the complete bounded one-shot contract.

An implementation with a known layer mapping, index, cache, or state-lifetime
error fails here even if an aggregate score happens to look acceptable.

## 3. Implementation parity

Use two references for different questions:

| Reference | Question |
|---|---|
| Same-dtype reference | Does the RPU implementation preserve the semantics of the chosen FP16 or quantized execution path? |
| CPU FP32 anchor | How much accuracy is lost relative to the higher-precision model? |

“Same dtype” is not a complete identity. Pin the reference device/backend,
framework and library versions, accumulation dtype, fusion settings, model
mode, deterministic settings, and input hashes. If any of these change, treat
the result as a different reference until equivalence is demonstrated.

If top-level parity or task evidence fails, locate the first divergent operator
or component before changing the runtime. Per-layer comparison is a diagnostic
escalation, not a mandatory release sweep for every passing model. For
high-dimensional tensors, report a distribution rather than only one extreme
value:

- mean, a low percentile, and minimum row cosine;
- relative L2 error and maximum absolute error;
- top-1 or top-k agreement for consumed logits; and
- the count and location of non-finite values or semantic mismatches.

Zero or nearly constant rows need absolute and relative tolerances instead of
cosine. Image-placeholder logits, padding rows, and other values that are not
consumed by the public output must be reported separately from consumed text,
decode, feature, or action outputs.

There is deliberately no repository-wide per-row cosine pass value. Before
running the release candidate, each profile freezes:

```text
reference identity: <backend, versions, dtype, accumulation and fusion>
consumed tensors:   <names and row/token selection>
parity metrics:     <cosine distribution, relative-L2, max-abs, agreement>
thresholds:         <value for each metric and treatment of zero/near-ties>
rationale:          <reference variability and downstream sensitivity>
```

Thresholds may be stricter for an operator than for a complete low-precision
model. They must not be relaxed after seeing a failing candidate without a
reviewed explanation and a new evidence version. If a family has no calibrated
contract, use the metric distribution to locate errors and keep the profile at
Experimental or Source-only until a task-linked contract is approved.

Quantized profiles use two comparisons: RPU against the exact quantized
same-dtype reference for implementation parity, then the quantized result
against the FP16 or FP32 anchor for retained task quality.

## 4. Task and end-to-end quality

`Supported` and `Limited` profiles require a frozen, representative evaluation
set and a metric that reflects the public output:

| Family | Examples of primary evidence |
|---|---|
| Causal LM | perplexity or task accuracy, teacher-forced token agreement, and fixed-prompt generation |
| VLM | text task quality plus image-conditioned and multi-image cases |
| Vision encoder | downstream retrieval, classification, or checkpoint-owned feature metric |
| VLA | normalized action error or action agreement plus the checkpoint-owned offline or closed-loop benchmark |

The profile freezes its task-quality target before the candidate run. A
retention target such as 99%, or 99.9% for a separately named high-accuracy
profile, can be appropriate for some accuracy metrics; it is an example rather
than a repository default. Other tasks may require a different relative target
or an absolute error bound. For a positive higher-is-better metric, quality
retention is `candidate / reference`; for a positive lower-is-better metric it
is `reference / candidate`. Metrics near zero, signed metrics, safety limits,
and action values with physical units require an absolute bound instead of a
ratio.

Exact greedy-token agreement is useful evidence, but a numerically stable
near-tie need not fail the model when the frozen task metric passes. Every
non-exact result must still be counted and explained; it cannot be silently
converted into an exact pass. VLA numerical validation does not certify robot
safety, coordinate frames, actuator limits, or deployment readiness.

## 5. Status decisions

| Status | Validation meaning |
|---|---|
| Supported | Every applicable hard, parity, task, lifecycle, and maximum-envelope gate passes for the exact public profile |
| Limited | The same quality bar as Supported passes, but the promised envelope is explicitly narrower |
| Experimental | Hard semantic, required finite-output, declared-repeatability, and safe rejection gates pass; parity or representative task evidence may still be incomplete |
| Component-only | The named component passes its component contract; no complete product-model claim is made |
| Source-only | Source is available without a runnable or quality commitment; known unsupported paths reject safely where applicable |
| Unsupported | No runnable public contract is offered for the profile |

A model-specific validation failure normally blocks only that profile's
support claim, asset, or example. It blocks the source release only when a
shared path has an uncontained semantic error, data corruption, non-finite
output, unsafe mutation, or cannot reject the affected profile before use.

## 6. Minimum release record

The signed evidence for a Supported or Limited profile contains:

1. the complete evidence identity from section 1;
2. hard-gate results and the maximum admitted envelope;
3. same-dtype and FP32-anchor metric distributions;
4. task metric, reference value, candidate value, and retention calculation;
5. warmup, repeatability, Graph lifecycle, and multi-input results; and
6. every exception, unrun gate, and owner approval.

Smoke execution and profiler traces are diagnostic evidence only. Unrun gates
remain pending; they are not inferred from another model size or a smaller
input.

For background, see PyTorch's notes on
[numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
and [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
[MLCommons Inference](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
illustrates task-specific relative and absolute quality targets rather than a
single tensor-similarity rule; RhinoForge likewise requires each profile to
choose and justify its task-appropriate metric and target.
