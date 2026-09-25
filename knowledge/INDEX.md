# RhinoForge knowledge index

Use this directory for concise, source-linked guidance. Public documentation
and source remain authoritative; see [the schema](SCHEMA.md) and
[maintenance workflow](WORKFLOWS.md).

## Canonical public documents

- [Architecture](../docs/architecture.md): runtime layers, FusedModelBase, SPM,
  Graph/batch execution, and DMA ownership.
- [API reference](../docs/api_reference.md): supported Python and `torch.rpu`
  surfaces.
- [Examples](../docs/model_support.md): example entry points, configuration layout, and runtime admission boundaries.
- [Model porting](../docs/model_porting.md): capability classification,
  implementation path, and verification gates.
- [Capability review](../docs/model_porting_capability_review.md): six-phase
  pre-code assessment and five possible outcomes.
- [Decoder step-by-step](../docs/new_model_step_by_step.md): minimal
  Adapter-only CausalLM checklist.
- [Porting pitfalls](../docs/pitfalls.md): focused correctness triage.
- [Getting started](../docs/getting_started.md): installation, metadata checks
  without device access, and first inference.
- [Restricted runtime assets](../docs/runtime_assets.md): matching external
  runtime packages, provider integrity checks, and installation.
- [Model execution and profiling](../docs/model_testing.md): direct examples,
  TOML runner, and profiling.
- [Model validation policy](../docs/validation_policy.md): proportionate host and hardware checks, numerical records and stop conditions.
- [Runtime configuration](../docs/runtime_config.md): supported controls,
  defaults, lifecycle, and risk.
- [Performance measurement](../docs/performance.md): correctness-first,
  reproducible latency and throughput reporting.
- [Model assets](../docs/model_assets.md) and
  [quantization](../docs/quantization.md): checkpoint records and offline
  conversion.

## Concepts

- [SPM allocation](concepts/spm-allocation.md): buffer lifetimes, planning, and
  capacity admission and standalone layout rebuilding.
- [GraphCache capture](concepts/graphcache-capture.md): BUILD/REPLAY lifecycle,
  signature correctness, policy inheritance, and failure cleanup.
- [DDR/SPM DMA wrappers](concepts/ddr-dma-wrappers.md): fixed, mutable, and
  immediate address ownership.
- [KV cache](concepts/kv-cache.md): sizing, position, transactional prefix
  replacement, and model state.
- [Weight swizzle](concepts/weight-swizzle.md): exactly-once layout conversion.
- [Multi-core broadcast](concepts/multicore-broadcast.md): safe immediate and
  batched launch behavior.
- [Multimodal rotary positions](concepts/mrope.md): three-axis position input,
  section, padded position slots, Graph contracts, and inference-tensor keepalive refresh.
- [Attention layout boundaries](concepts/sdpa-layout.md): unified and PyTorch
  layout conversion without exposing cache storage internals.
- [Host launch runtime boundary](concepts/launch-runtime.md): restricted library
  dependency, public wrapper contract, queue ownership, and buffer lifetime.
- [Runtime profiles](concepts/runtime-profiles.md): profile identity,
  configuration precedence, cold owner binding, host settings, lifecycle, and safe change classification.
- [Chunk planning and admitted envelopes](concepts/chunk-plan-and-envelope.md):
  logical versus physical lengths, planning feasibility, padding ownership,
  decode override semantics, and profile admission.
- [Three-stage chunk execution](concepts/three-stage-chunk-plan.md): shared
  input/QKV/compute schedules, semantic spans, exact plan identity, and
  attention-storage fallback.
- [Multi-component ownership and handoff](concepts/component-handoff.md):
  Graph, SPM, DMA, output, HYViT input admission, and teardown boundaries between
  model components.
- [Automatic residual reduction](concepts/allreduce-routing.md): one generated
  eight-core ring route, partial-producer preparation, and Graph ownership.
- [Generated Linear auto-tiling](concepts/linear-autotiling.md): shared FP16,
  W8A16, and packed-W4 shape planning without profile tile selectors.
- [Lazy adapter registry](concepts/lazy-adapter-registry.md): built-in manifest,
  on-demand imports, and trusted external plugin ownership.
- [Deferred DMA tensor lifetime](concepts/deferred-dma-lifetime.md): keeping
  source and destination storage alive across delayed BUILD and REPLAY.
- [Backend registration boundary](concepts/backend-registration-boundary.md):
  review fragments, one translation unit, and lifecycle verification.
- [Live model and policy lifecycle](concepts/live-instance-lifecycle.md):
  transactional installation, one process owner, failure cleanup, and teardown.

## Task guides

- [Pitfalls overview](synthesis/pitfalls-overview.md): symptom-to-contract
  debugging map.
- [Porting playbook](synthesis/porting-playbook.md): shortest path from profile
  assessment to release evidence.
- [Model family implementation map](synthesis/model-family-map.md): public APIs,
  adapter owners, and runtime contracts for Qwen3, Qwen3-VL, Pi0.5, RhinoVLA,
  and the other indexed families, including Pi0.5 fixed-noise comparisons and
  cold Graph resource planning.
- [Debugging and profiling playbook](synthesis/debugging-playbook.md): bounded
  numerical triage, pointer-free Graph structure, profiler choice, and stop
  conditions.
- [`rhinoforge-port`](../.agents/skills/rhinoforge-port/SKILL.md): AI workflow
  for capability assessment and model-port implementation.

## Knowledge operations

- [Retrieval](RETRIEVAL.md): Markdown-first lookup and optional semantic-index
  boundary.
- [Saved queries](queries/README.md): reusable, source-linked decision records.
- [External snapshots](raw/README.md): provenance and sanitization rules for
  reviewed public sources.
