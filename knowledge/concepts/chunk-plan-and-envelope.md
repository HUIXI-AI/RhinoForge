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
- **Installed capacity** is the cold-bound logical storage/position capacity
  requested for one model. A cache must fit it; aligned physical storage also
  includes any required padding.
- **Chunk size** bounds the rows handled by one planned unit. It is `"auto"` or
  a positive multiple of 16 on the public configuration surface.
- **Padding budget** limits optional padding considered by a planner;
  **padding rows** requests an automatic or exact amount. An exact value and a
  budget are alternatives.
- **Admitted envelope** is the complete set of logical and physical shapes for
  which the profile has correctness, lifecycle, and resource evidence.

The adapter must return logical outputs and keep logical cache position even
when the physical plan uses more rows.

KV allocation and per-chunk feasibility are independent. Supported adapters
bind planning length bounds to installed capacity and physical padding instead
of a fixed historical token limit. Chunk ceilings, SPM usage, operator/register
and coordinate widths, alignment and semantic input shapes still constrain
execution. The allocator either provides the requested storage or reports
failure; it does not silently shorten it. Default capacity is not an allocation
ceiling, and a larger successful allocation does not extend hardware-validation
evidence. See [KV cache](kv-cache.md) for capacity entry points.

## Planning contract

1. Reject a model or request outside the exact public profile before loading or
   transforming weights.
2. Bind cold `rpu_execution` fields before device installation. Use
   `reconfigure_rpu_execution` between requests for session-supported planning
   fields; it replaces stages and retires affected Graphs. Core count, precision
   and installation-capacity changes require a new model or policy.
3. Evaluate only deterministic candidates. A fused subsystem's
   `declare_buffers` and chunk-validity hooks must not launch work or depend on
   mutable runtime state.
4. Treat SPM capacity, alignment, operator/coordinate widths, semantic input
   shapes, actual cache/position capacity, and model-specific constraints as
   hard feasibility gates. A profile's chunk ceiling does not impose a total
   KV-capacity ceiling.
5. Let the generic planner choose an unpadded feasible chunk. If an adapter
   admits logical padding, it owns the search, semantic inputs, output slicing,
   and cache rewind.
6. Include the resolved physical shape and every value that changes execution
   meaning in the Graph signature or a validated mutable update path.
7. Record the requested and resolved plan with the exact release profile. Do
   not promote an adjacent length or model size from one passing example.

The legacy native prefill override clamps a single-token decode to a capacity
of 16 without rewriting the stored prefill value; multi-token requests keep
strict equality. This compatibility behavior is not an additional public
`rpu_execution` tuning rule. [Fused planning implementation](../../src/core/fused_model_base.cpp)

## Exact requests and semantic inputs

A public integer `chunk_size` fixes the Q/K/V and compute capacities. It does
not rewrite the input schedule or the semantic spans that separate images,
views or independent streams. Those spans belong to the input owner. A model
may require stricter alignment than the common 16-row boundary.

Without a separately installed cost profile, the shared planner ranks admitted
plans by the largest stage chunk count, optional padding, largest final-chunk
deficit, then stable capacity tie-breaks. It uses actual input/QKV/compute
schedules; this is deterministic structural selection.

Automatic search considers only bounded, admitted candidates. Exhausting a
search limit is a rejection, not permission to truncate the domain and report
a complete result. An exact request that has no matching physical plan fails
without substituting a nearby size. Unexpected device or programming errors
remain errors; they are not silently treated as infeasible candidates.

The public mapping supports the entry point's `prefill`, `vision` and `action`
stages. Internal request modes, semantic spans and fixed decode controls are
not additional tuning fields. Component overrides take precedence over stage
defaults, and each component resolves against its own envelope.

## Stop conditions

Stop before execution when no candidate fits the resource envelope, an exact
padding request conflicts with its budget, a requested stage or field is not
supported by the entry point, or the adapter cannot preserve logical output and
cache semantics. A fallback that merely completes the call does not widen the
profile.

## Sources

- [Runtime configuration: TOML and `rpu_execution`](../../docs/runtime_config.md#configuration-and-execution)
- [Architecture: planning and invalidation](../../docs/architecture.md#planning-and-invalidation-lifecycle)
- [Architecture: SPM design](../../docs/architecture.md#spm-design)
- [Examples](../../docs/model_support.md)
- [Public execution-configuration validator](../../python/rpu_backend/api/_execution.py)
- [Fused model planning contract](../../src/core/fused_model_base.h)
