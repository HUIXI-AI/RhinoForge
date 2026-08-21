# Model porting capability review

Run this review before writing a model adapter. Its purpose is to decide
whether an exact model profile fits RhinoForge's current public interfaces and
to expose blockers before checkpoint loading or irreversible weight changes.

Read [Model support](model_support.md), [Architecture](architecture.md), and
[Model porting](model_porting.md) first. Review one exact profile at a time; a
family name or matching Hugging Face architecture is not a profile.

## Review record

Record these fields in the port proposal:

- model source, exact revision, configuration hash, and
  `config.architectures[0]`;
- hidden, intermediate, layer, attention-head, KV-head, and vision geometry;
- precision and quantization metadata;
- batch, sequence, cache, image, video, and action-horizon limits;
- preprocessing and returned-output contract;
- RhinoForge, PyTorch, Rhino Launch, and combined operator-asset versions; and
- one final outcome from the five defined below.

An unknown value is a review gap, not permission to inherit a nearby profile.

## Phase 1: freeze the profile

Compare the target with every row and exclusion in
[Model support](model_support.md). Define the smallest input envelope that is
useful and testable. Include every option that changes model math, tensor
shape, Graph topology, cache layout, or precision.

Fail the review early when any of these is true:

- the exact profile is explicitly unsupported;
- the required precision is unavailable;
- the target needs training or fine-tuning;
- required model code cannot run with `trust_remote_code=False`; or
- the checkpoint, preprocessing, or output contract cannot be pinned.

**Done when:** two reviewers can select the same checkpoint and construct the
same input and expected output from the record alone.

## Phase 2: trace the reference forward

Run the upstream model in evaluation mode. Freeze both the exact same-dtype
reference executable and a CPU FP32 anchor as defined by
[Model validation policy](validation_policy.md). Trace one complete request
from preprocessing to returned output and record, in execution order:

- module or mathematical operation;
- input and output shapes and dtypes;
- tensor layout changes;
- masks, position data, lookup tables, and other semantic inputs;
- cache or recurrent-state reads and writes; and
- the same-dtype value used for implementation parity and the CPU FP32 value
  used as the higher-precision anchor.

Cover prefill and decode separately for a language model. For a vision or VLA
model, include every component boundary and at least one multi-input case.
Do not replace missing model math with an approximation merely to complete the
trace.

**Done when:** every returned value and persistent state update has an upstream
producer in the trace.

## Phase 3: map the trace to public capabilities

Map every reference step to the smallest existing RhinoForge path:

| Mapping | Evidence required |
|---|---|
| Existing adapter | The exact profile and input envelope already pass its preflight |
| Shared causal decoder | The target math matches the `llama` or `qwen3` branch used by the checked-in template |
| Existing RPU operation | Its public schema covers the exact shapes, dtypes, layouts, and state update |
| Checked CPU fallback | Correctness is preserved, but the fallback is listed explicitly and is not treated as RPU support evidence |
| New host operation | Mathematical definition, tensor contract, validation, registration, and release-matched operator availability are all specified |
| New fused subsystem | Repeated execution needs a new `FusedModelBase` SPM manifest and layer subgraph |

Inspect the adapter template under
`python/rpu_backend/adapters/_template/`, the public registrations in
`src/core/rpu_dispatch_registrations.inc`, and the fused implementations under
`src/fused/`. Source-file presence alone is not a capability claim.

For each gap, write one of: reuse, adapter change, runtime change, asset change,
or blocker. A required operation absent from the release-matched asset is a
blocker until that asset is updated and validated.

**Done when:** every Phase 2 row has exactly one mapping and an owner.

## Phase 4: prove resource and lifecycle feasibility

Check the complete profile against the public runtime contract:

- eight cores, 8 MiB SPM per core, and `SPM_PLANNING_BUDGET` at 99% of usable
  SPM (about 7.9 MiB per core);
- FP16 as the primary dtype and 16-element alignment where required;
- deterministic, alias-aware `BufferDecl` planning for a fused subsystem;
- replay-persistent SPM data declared as `Persistent` or
  `PersistentPerLayer`;
- fixed DMA only for storage with a stable address, mutable DMA for caller
  inputs and fresh outputs;
- every semantic input represented in the `GraphSignature` or refreshed by a
  validated mutable transfer;
- independent storage for Python-visible outputs retained across calls; and
- exactly-once weight transformation and the process ownership limits in
  [Architecture](architecture.md).

SPM capacity is a hard admission gate. A smaller passing shape does not prove
the maximum envelope. If a shape cannot be planned safely, narrow the profile
or mark it blocked.

**Done when:** the maximum admitted input has a feasible memory plan and an
explicit Graph, DMA, cache, and output-lifetime contract.

## Phase 5: define verification before implementation

Write the gates that the port must pass:

1. unsupported configurations fail before weight loading or transformation;
2. the adapter registry and execution configuration pass board-free tests;
3. the exact profile defines the hard semantic, same-dtype parity, FP32-anchor,
   and representative task gates required by
   [Model validation policy](validation_policy.md);
4. `RPU_WARMUP=0`, `1`, and `3` produce bit-identical outputs;
5. a repeated signature performs one BUILD followed by stable REPLAY, with a
   fixed cache size and `cache_invariant_ok()` remaining true;
6. different semantic inputs with the same shape match the uncaptured path;
7. multiple inputs, and multiple images where applicable, do not alias returned
   storage; and
8. the maximum admitted envelope and the public TOML example run in a clean
   process.

Use [Model execution and profiling](model_testing.md) for runnable entry points.
Profiling output is diagnostic evidence, not a numerical gate.

**Done when:** every gate names an input, reference, expected result, and place
to record the result.

## Phase 6: issue one outcome

| Outcome | Decision |
|---|---|
| Existing path | No port is needed; use the already supported exact profile |
| Adapter-only | Existing public operations and runtime cover the trace |
| Runtime extension | A shared host-side operation or runtime behavior is missing |
| New fused subsystem | A new repeated SPM-resident execution path is required |
| Blocked | An operation, asset, shape, precision, memory plan, dependency, or numerical gate is unavailable |

List assumptions and blockers beside the outcome. Continue with
[New model step by step](new_model_step_by_step.md) only for an approved
decoder-only Adapter-only result. An Existing path needs no port; use its
already supported exact profile. Use the broader workflow in [Model
porting](model_porting.md) for runtime extensions, fused subsystems, vision
models, and multi-component policies.

Do not add a support-matrix row until all profile gates have run against one
immutable source, dependency, checkpoint, configuration, and asset set.
