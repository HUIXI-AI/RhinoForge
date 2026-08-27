# Chunk planning and admitted envelopes

Chunk planning chooses a physical execution shape for one logical input. A
successful plan is not by itself a support claim: the checkpoint, precision,
input range, runtime configuration, and asset set must also belong to an
admitted profile.

## Distinguish the lengths

- **Logical length** is the caller-visible number of tokens, image rows, or
  action rows.
- **Execution length** includes any adapter-owned padding needed by the exact
  profile.
- **Chunk size** bounds the rows handled by one planned unit. It is `"auto"` or
  a positive multiple of 16 on the public configuration surface.
- **Padding budget** limits optional padding considered by a planner;
  **padding rows** requests an automatic or exact amount. An exact value and a
  budget are alternatives.
- **Admitted envelope** is the complete set of logical and physical shapes for
  which the profile has correctness, lifecycle, and resource evidence.

The adapter must return logical outputs and keep logical cache position even
when the physical plan uses more rows.

## Planning contract

1. Reject a model or request outside the exact public profile before loading or
   transforming weights.
2. Bind `rpu_execution` before RPU installation and the first Graph BUILD. A
   later change requires a new model or policy.
3. Evaluate only deterministic candidates. A fused subsystem's
   `declare_buffers` and chunk-validity hooks must not launch work or depend on
   mutable runtime state.
4. Treat SPM capacity, alignment, operator shape limits, cache capacity, and
   model-specific constraints as hard feasibility gates.
5. Let the generic planner choose an unpadded feasible chunk. If an adapter
   admits logical padding, it owns the search, semantic inputs, output slicing,
   and cache rewind.
6. Include the resolved physical shape and every value that changes execution
   meaning in the Graph signature or a validated mutable update path.
7. Record the requested and resolved plan with the exact release profile. Do
   not promote an adjacent length or model size from one passing example.

## Stop conditions

Stop before execution when no candidate fits the resource envelope, an exact
padding request conflicts with its budget, a requested stage or field is not
supported by the entry point, or the adapter cannot preserve logical output and
cache semantics. A fallback that merely completes the call does not widen the
profile.

## Sources

- [Runtime configuration: TOML and `rpu_execution`](../../docs/runtime_config.md#toml-parameter-catalog)
- [Architecture: planning and invalidation](../../docs/architecture.md#planning-and-invalidation-lifecycle)
- [Architecture: SPM design](../../docs/architecture.md#spm-design)
- [Model support](../../docs/model_support.md)
- [Public execution-configuration validator](../../python/rpu_backend/api/_execution.py)
- [Fused model planning contract](../../src/core/fused_model_base.h)
