# RhinoForge knowledge index

Use this directory for concise, source-linked guidance. Public documentation
and source remain authoritative; see [the schema](SCHEMA.md) and
[maintenance workflow](WORKFLOWS.md).

## Canonical public documents

- [Architecture](../docs/architecture.md): runtime layers, FusedModelBase, SPM,
  Graph/batch execution, and DMA ownership.
- [API reference](../docs/api_reference.md): supported Python and `torch.rpu`
  surfaces.
- [Model support](../docs/model_support.md): exact public profile status and
  limits.
- [Model porting](../docs/model_porting.md): capability classification,
  implementation path, and verification gates.
- [Capability review](../docs/model_porting_capability_review.md): six-phase
  pre-code assessment and five possible outcomes.
- [Decoder step-by-step](../docs/new_model_step_by_step.md): minimal
  Adapter-only CausalLM checklist.
- [Porting pitfalls](../docs/pitfalls.md): focused correctness triage.
- [Getting started](../docs/getting_started.md): installation and first
  inference.
- [Restricted runtime assets](../docs/runtime_assets.md): authorized Release
  page, compatibility set, verification, installation, and revocation.
- [Model execution and profiling](../docs/model_testing.md): direct examples,
  TOML runner, and profiling.
- [Model validation policy](../docs/validation_policy.md): independent hard
  semantic, implementation-parity, task-quality, and lifecycle gates.
- [Runtime configuration](../docs/runtime_config.md): supported controls,
  defaults, lifecycle, and risk.
- [Performance measurement](../docs/performance.md): correctness-first,
  reproducible latency and throughput reporting.
- [Model assets](../docs/model_assets.md) and
  [quantization](../docs/quantization.md): checkpoint records and offline
  conversion.

## Concepts

- [SPM allocation](concepts/spm-allocation.md): buffer lifetimes, planning, and
  capacity admission.
- [GraphCache capture](concepts/graphcache-capture.md): BUILD/REPLAY lifecycle
  and signature correctness.
- [DDR/SPM DMA wrappers](concepts/ddr-dma-wrappers.md): fixed, mutable, and
  immediate address ownership.
- [KV cache](concepts/kv-cache.md): sizing, position, and model state.
- [Weight swizzle](concepts/weight-swizzle.md): exactly-once layout conversion.
- [Multi-core broadcast](concepts/multicore-broadcast.md): safe immediate and
  batched launch behavior.
- [Multimodal rotary positions](concepts/mrope.md): three-axis position input,
  section, padding, and Graph contracts.
- [Attention layout boundaries](concepts/sdpa-layout.md): unified and PyTorch
  layout conversion without exposing cache storage internals.
- [Host launch runtime boundary](concepts/launch-runtime.md): restricted library
  dependency, public wrapper contract, queue ownership, and hardware profiling.
- [Runtime profiles](concepts/runtime-profiles.md): profile identity,
  configuration precedence, lifecycle, and safe change classification.

## Task guides

- [Pitfalls overview](synthesis/pitfalls-overview.md): symptom-to-contract
  debugging map.
- [Porting playbook](synthesis/porting-playbook.md): shortest path from profile
  assessment to release evidence.
- [Model family implementation map](synthesis/model-family-map.md): public APIs,
  adapter owners, and runtime contracts for Qwen3, Qwen3-VL, Pi0.5, RhinoVLA,
  and the other indexed families.
- [Debugging and profiling playbook](synthesis/debugging-playbook.md): bounded
  numerical triage, Graph evidence, profiler choice, and stop conditions.
- [`rhinoforge-port`](../.agents/skills/rhinoforge-port/SKILL.md): AI workflow
  for capability assessment and model-port implementation.

## Knowledge operations

- [Retrieval](RETRIEVAL.md): Markdown-first lookup and optional semantic-index
  boundary.
- [Saved queries](queries/README.md): reusable, source-linked decision records.
- [External snapshots](raw/README.md): provenance and sanitization rules for
  reviewed public sources.

See [log.md](log.md) for public knowledge-base maintenance entries.
