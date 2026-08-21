# RhinoForge agent guide

RhinoForge is an inference-only PyTorch `PrivateUse1` backend exposed as
`torch.rpu`. Its distribution name is `rhinoforge`; its Python package remains
`rpu_backend`.

## Read before changing code

1. Read [model support](docs/model_support.md) for the exact public profile.
2. Read [architecture](docs/architecture.md) for runtime ownership and execution
   contracts.
3. Read [API reference](docs/api_reference.md) before changing a public surface.
4. Use [model validation](docs/validation_policy.md) for independent hard,
   same-dtype parity, FP32-anchor, task-quality, and lifecycle gates.
5. For model work, start with the
   [capability review](docs/model_porting_capability_review.md), then follow
   [model porting](docs/model_porting.md), the decoder
   [step-by-step checklist](docs/new_model_step_by_step.md), and the
   [`rhinoforge-port` skill](.agents/skills/rhinoforge-port/SKILL.md) as
   applicable. Use [porting pitfalls](docs/pitfalls.md) before adding a
   workaround.
6. Query [the knowledge index](knowledge/INDEX.md) before searching source for a
   documented concept.

Use [getting started](docs/getting_started.md) and
[restricted runtime assets](docs/runtime_assets.md) for installation,
[model testing](docs/model_testing.md) for runnable entry points and profiling,
and [runtime configuration](docs/runtime_config.md) for supported controls.

## Required engineering contracts

- Treat support as an exact checkpoint, precision, input-envelope, execution,
  launch-library, and operator-asset profile. Source presence, an architecture
  match, or a registry alias is not a support claim.
- Reject unsupported profiles before full weight loading or irreversible
  transformation.
- Transform a model's weight layout exactly once. Reload a clean model after a
  partial transformation or failed installation.
- Respect process ownership: the main fused CausalLM path permits one live
  RPU-resident model or policy per process.
- Keep `FusedModelBase::declare_buffers` deterministic and free of launch side
  effects. Declare state that must survive Graph replay as persistent.
- For a repeated Graph signature, prove one BUILD followed by stable REPLAY:
  replay count grows, cache size stays fixed, and `cache_invariant_ok()` remains
  true. Treat a bounded one-shot path as an explicit model-contract exception.
- Include every semantic input in the Graph signature or update it through the
  documented mutable DMA path.
- Use fixed DMA only for addresses stable across replay. Give Python-visible
  outputs independent storage when callers may retain them.
- Route multi-core work through the framework batch/Graph path or explicitly
  enable host-parameter broadcast for an immediate launch.
- Keep device-program implementation details and opaque operator-asset formats
  outside this repository. Public host wrappers may describe tensor metadata,
  memory locations, synchronization, core selection, and parameter packing.
- Treat exported activations and low-level hardware trace artifacts as
  sensitive application data.

## Change discipline

- Make the smallest change that satisfies a verified profile; reuse public
  helpers before adding abstractions or dependencies.
- Keep profile guards, cache ownership, teardown, and error paths explicit.
- Run board-free checks before board validation. A port is not complete until
  the exact profile passes the numerical, warmup, Graph-lifecycle, input-
  envelope, and end-to-end gates in [model porting](docs/model_porting.md).
- Update public documentation and [the knowledge workflow](knowledge/WORKFLOWS.md)
  when a public contract changes.
