---
name: rhinoforge-port
description: Port or assess a Hugging Face inference model for the RhinoForge RPU PyTorch backend. Use for model capability reviews, adapter work, profile admission, operator-gap classification, Graph/SPM/DMA planning, and release-profile verification.
---

# Assess or port a model to RhinoForge

## Choose one mode

Neither mode performs hardware, privileged, network, branch, push, or publish
actions without explicit user authorization. If required hardware or assets are
unavailable, record the affected gates as pending.

- **Assess:** perform the six-phase
  [capability review](../../../docs/model_porting_capability_review.md), issue
  one of its five outcomes, record whether the evidence is certified, and make
  no code or repository changes. Read
  [the assessment reference](references/assessment.md) and
  [public capability map](references/capability-map.md) before starting.
- **Port:** require an accepted assessment first. For an Adapter-only
  decoder CausalLM, follow the
  [step-by-step checklist](../../../docs/new_model_step_by_step.md). For a
  runtime extension, fused subsystem, vision model, or policy, follow the
  broader [model-porting guide](../../../docs/model_porting.md). An Existing
  path needs no code change; stop when the outcome is Existing path or Blocked.

Load only the family reference that matches the target:

- [Decoder CausalLM](references/families/causal-lm.md)
- [Vision encoder](references/families/vision.md)
- [Vision-language model](references/families/vlm.md)
- [Vision-language-action policy](references/families/vla.md)

## Establish the contract

1. Read [model support](../../../docs/model_support.md),
   [model porting](../../../docs/model_porting.md), and
   [model validation](../../../docs/validation_policy.md), then
   [architecture](../../../docs/architecture.md).
2. Define one exact profile: architecture, checkpoint revision, precision,
   input envelope, execution settings, launch-library version, and operator-
   asset version.
3. Trace the exact same-dtype reference and CPU FP32 anchor from preprocessing
   to returned outputs. Record tensor shapes, dtypes, state updates, cache
   behavior, numerical outputs, and the complete reference identities.
4. Map each operation to an existing public adapter, operator, Graph, SPM, DMA,
   or fallback path.

Classify the result as Existing path, Adapter-only, Runtime extension, New
fused subsystem, or Blocked. Choose the smallest class that preserves the
reference math. Stop as Blocked when a required operation, asset, precision,
memory plan, or numerical gate is unavailable.

## Implement the smallest valid path

For a decoder-only CausalLM, start from
`python/rpu_backend/adapters/_template/minimal_causal_lm.py`; use
`adapter.py` only when the port needs the annotated structure.

1. Add an exact preflight guard that runs before full weight loading or
   irreversible transformation.
2. Reuse the shared decoder when its math matches. Keep configuration,
   checkpoint loading, weight conversion, cache ownership, Graph signatures,
   orchestration, and teardown in Python.
3. Add a host-side runtime extension only when existing public operations
   cannot express the required behavior.
4. Add a `FusedModelBase` subsystem only for a repeated fused path that needs
   its own SPM plan. Keep buffer declaration deterministic, declare replay-
   persistent state, and invalidate model state after layout-affecting setters.
5. Choose fixed, mutable, or immediate DMA from address ownership and replay
   lifetime. Give caller-retained outputs fresh storage.
6. Register an in-tree adapter in
   `python/rpu_backend/adapters/_manifest.py` and through
   `register_adapter`. Use the `rpu_backend.plugins` entry point for an external
   package; never replace a built-in binding.

Do not duplicate a public wrapper, broaden a support family from one passing
size, transform one model instance twice, or approximate missing model math.
Keep restricted implementation details and opaque asset formats out of the
change.

## Verify before claiming support

Run the board-free checks and, when explicitly authorized and available, the
exact-board gates in [model porting](../../../docs/model_porting.md#8-verification-gates).
At minimum,
verify early rejection, configuration validation, clean build/import, the
profile's hard semantic, same-dtype, FP32-anchor, and task-quality gates,
warmup stability, the applicable Graph lifecycle, multiple inputs, the maximum
admitted envelope, and the public end-to-end entry point.

Record exact revisions, hashes, versions, precision, envelope, configuration,
and results. Update [model support](../../../docs/model_support.md) only after
the evidence matches the release profile. Use
[porting pitfalls](../../../docs/pitfalls.md) when a check fails and the
[public playbook](../../../knowledge/synthesis/porting-playbook.md) for focused
execution concepts.
