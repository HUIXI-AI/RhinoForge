# RhinoForge knowledge index

Use this directory for concise, source-linked guidance. Public documentation
and source remain authoritative; see [the schema](SCHEMA.md) and
[maintenance workflow](WORKFLOWS.md).

## Canonical public documents

- [v1.2.0 upgrade guide](../docs/upgrading_1_2.md): installation,
  current example paths, required inputs and runtime settings.

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
- [Install, run and develop](../docs/getting_started.md): Miniconda3, the single
  `RHINOFORGE_ENV`, dependency bundles, source/editable/wheel builds,
  metadata checks without device access, and first inference.
- [Dependency environments](../docs/environment_bundle.md): public dependency-only
  archives, independent versions, source binding and the boundary with platform assets.
- [Restricted runtime assets](../docs/runtime_assets.md): matching external
  runtime packages, provider integrity checks, and installation.
- [Run and debug models](../docs/model_testing.md): direct examples,
  TOML runner, and profiling.
- [Model validation policy](../docs/validation_policy.md): proportionate host and hardware checks, runtime contracts and stop conditions.
- [Runtime configuration](../docs/runtime_config.md): supported controls,
  defaults, lifecycle, and risk.
- [Compute-core topology](../docs/planner_core_topology.md): cold core budgets,
  component allocation, physical padding and matching cache construction.
- [Model assets](../docs/model_assets.md) and
  [quantization](../docs/quantization.md): checkpoint records and offline
  conversion.

- [Release policy](../docs/release_policy.md): development and release branches,
  independent component versions, dependency locks and immutable artifacts.

## Concepts

- [SPM allocation](concepts/spm-allocation.md): buffer lifetimes, planning, and
  capacity admission and standalone layout rebuilding.
- [GraphCache capture](concepts/graphcache-capture.md): BUILD/REPLAY lifecycle,
  signature correctness, policy inheritance, and failure cleanup.
- [DDR/SPM DMA wrappers](concepts/ddr-dma-wrappers.md): fixed, mutable, and
  immediate address ownership.
- [R1 DDR bandwidth limits](concepts/ddr-bandwidth.md): SMMU/controller read
  ceilings, 4/6/8-core physical core sets, and decode bandwidth floors.
- [KV cache](concepts/kv-cache.md): sizing, position, transactional prefix
  replacement, and model state.
- [Weight swizzle](concepts/weight-swizzle.md): exactly-once layout conversion.
- [Multi-core broadcast](concepts/multicore-broadcast.md): safe immediate and
  batched launch behavior.
- [Multimodal rotary positions](concepts/mrope.md): three-axis position input,
  section, padded position slots, Graph contracts, and inference-tensor keepalive refresh.
- [Attention layout boundaries](concepts/sdpa-layout.md): unified and PyTorch
  layout conversion, versioned CPU masks and replay lifetimes.
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
  Graph, SPM, DMA, CPU publication, output, HYViT input admission, and teardown boundaries between
  model components.
- [Automatic residual reduction](concepts/allreduce-routing.md): one generated
  eight-core ring route, partial-producer preparation, and Graph ownership.
- [Generated Linear auto-tiling](concepts/linear-autotiling.md): shared FP16,
  W8A16, and packed-W4 shape planning, shared MLP fusion and bounded weight reuse.
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
  and the other indexed families, including Pi0.5 fixed-noise comparisons,
  RhinoVLA v3 FP16/W8A16 RTC request semantics, and cold Graph resource planning.
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

## Speech inference

- [Qwen3 ASR/TTS](../docs/qwen3_audio.md): pinned assets, source preparation,
  input limits, CPU/device ownership, Predictor checked replay, one-shot output
  heads, 1–255 frame device codec frontend, 31-layer Conv1d + 6 transposed convolutions + SnakeBeta/residual/ConvNeXt RPU offload, and Python API calls.
