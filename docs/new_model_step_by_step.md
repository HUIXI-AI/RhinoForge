# New decoder model: step by step

This is the shortest supported path for a new decoder-only Hugging Face
CausalLM whose capability-review outcome is **Adapter-only**. It reuses the
shared causal decoder and does not define a new fused subsystem.

Complete [Model porting capability review](model_porting_capability_review.md)
first. If the model math does not match the existing `llama` or `qwen3`
decoder branch, stop here and use [Model porting](model_porting.md).

## 1. Pin the exact profile and numerical references

Record the checkpoint revision, configuration hash, precision, input envelope,
and expected output contract. Freeze an exact same-dtype reference executable
for implementation parity and a separate CPU FP32 model as the higher-precision
anchor. Pin the reference backend, dependency versions, accumulation and fusion
settings, call `eval()`, disable gradients, and save reference inputs and
outputs for:

- a multi-token prefill;
- at least one teacher-forced decode step;
- the maximum admitted cache length; and
- two different inputs with the same shapes.

Keep the CPU reference instance separate. RPU weight transformation is in place
and cannot be reversed safely.

**Done means:** the same pinned inputs reproduce both reference contracts
without RhinoForge.

## 2. Prove the shared decoder is mathematically exact

Compare the target forward with the checked-in Qwen3 and Llama adapters. Verify
all of these explicitly:

- normalization placement and epsilon;
- Q/K normalization presence;
- attention head, KV-head, head-dimension, mask, and cache behavior;
- rotary-position convention;
- projection bias and tied-weight behavior;
- MLP projections and activation; and
- logits and cache return semantics.

Choose `DECODER_ARCH = "qwen3"` only when the Q/K-normalized path matches;
choose `"llama"` only when the non-Q/K-normalized path matches. Similar class
names are not evidence.

**Done means:** every target operation maps to the selected decoder branch with
no omitted or approximated math.

## 3. Copy and fill the minimal adapter

```bash
cp python/rpu_backend/adapters/_template/minimal_causal_lm.py \
   python/rpu_backend/adapters/<model_name>.py
```

Start with the configuration block already present in the template:

- `HF_ARCH` must equal `config.architectures[0]` exactly;
- `DECODER_ARCH` is the branch proven in Step 2;
- `SUPPORTED_PROFILES` contains the admitted five-field geometry tuples;
- `RMSNORM_CLASS` and `ROTARY_CLASS` come from the installed upstream model
  implementation;
- `SKIP_LINEAR_NAMES` contains leaf child names, not qualified paths, and only
  for specialized linear modules transformed by another authoritative path;
  each name matches that child name in every layer;
- `_CHUNK_ENVELOPE` contains only measured-safe profile rows and no template
  example.

The five-field tuple is not the complete model contract. Extend `preflight()`
to reject every Step 2 configuration field that can change math or layout,
including the applicable normalization epsilon, rotary convention, bias,
activation, and tied-weight settings. Validate exact values before loading or
transforming weights.

Do not add a base class, factory, or second weight walker. The template provides
the geometry preflight, exactly-once transformation, model ownership, runtime
installation, and cleanup on failure; the copied adapter completes the exact
profile preflight.

**Done means:** importing the copied module is inert except for its one adapter
registration, and every unsupported geometry or math-affecting configuration
fails in `preflight()`.

## 4. Connect one registry route

For an in-tree adapter, add one tuple to
`python/rpu_backend/adapters/_manifest.py` and add its architecture to
`RPUModelForCausalLM._SUPPORTED_BUILTIN_ARCHITECTURES`. The module-level
`register()` call and manifest tuple must name the same architecture and class.

For an external distribution, leave the in-tree manifest unchanged and expose
one trusted `rpu_backend.plugins` entry point as documented in
[API reference](api_reference.md). A plugin cannot override a built-in binding.

Add one board-free check that uses configuration only and proves:

1. the exact profile resolves to the new adapter;
2. a neighboring size and a same-geometry profile with one changed Step 2 math
   field both fail before full checkpoint loading; and
3. malformed `rpu_execution` is rejected before model mutation.

**Done means:** `list_adapters()` reports one deterministic binding and an
unsupported profile cannot reach the weight loader.

## 5. Keep the installation order intact

Retain the template's `to_rpu()` sequence:

1. validate hardware attributes and claim the live instance;
2. mark the irreversible transformation as started;
3. install the idempotent class swaps;
4. call `swizzle_model_inplace(..., skip_names=SKIP_LINEAR_NAMES)` once;
5. move the transformed model to `rpu`;
6. install the shared causal decoder with the cold `rpu_execution` mapping;
7. validate the installed model; and
8. mark it ready only after every prior action succeeds.

Do not retry a failed transformation on the same instance. Reload a clean CPU
model. Do not add a process-global chunk setter; public loaders bind chunk and
padding choices through `rpu_execution`.

**Done means:** a second `.to("rpu")` is a safe no-op only for a fully installed
model, while partial installation and attempts to move transformed weights back
to CPU fail clearly.

## 6. Verify Graph, cache, and storage ownership

The shared decoder owns a per-model `GraphCache`; do not add another capture
layer. Confirm its signature covers every shape and semantic mode that can
change the emitted forward. Any same-shape semantic input not stored in the
signature must be refreshed through the existing mutable-transfer contract.

Run the same admitted signature repeatedly and check:

- the first call is BUILD and later calls are REPLAY;
- cache size remains fixed and replay count increases;
- `cache_invariant_ok()` stays true;
- a different semantic value produces the same bytes as the uncaptured path;
  and
- outputs retained by the caller have independent storage.

**Done means:** warmup does not change values, rebuild the same signature, or
overwrite an earlier returned output.

## 7. Add one public example and its checks

Reuse `examples/causal_lm.py` when its TOML schema already covers the model.
Add only a new file under `examples/configs/` for the exact profile; create a
new script only when the input or output contract is genuinely different.

The example must pass configuration validation without importing the native
backend:

```bash
python examples/causal_lm.py --config examples/configs/<profile>.toml --check-config
```

Update [Model assets](model_assets.md) with the source, revision, destination,
and hashes, and update [Model execution and profiling](model_testing.md) only
when the entry-point inventory changes. Keep release-only environment values
out of TOML; use [Runtime configuration](runtime_config.md) as the allowlist.

Run the existing board-free suite:

```bash
python -m pytest \
  tests/test_public_runtime.py \
  tests/test_configuration.py \
  tests/test_kernel_contracts.py
```

**Done means:** configuration-only checks, package import, registry behavior,
and the existing public contracts pass without a board.

## 8. Run the profile gates and promote it

Build from a clean process using the release-matched dependencies and asset,
then run the exact TOML on an RPU board. The required evidence is:

- the hard semantic, same-dtype implementation-parity, CPU FP32 anchor, and
  representative task gates frozen under
  [Model validation policy](validation_policy.md);
- bit-identical output for `RPU_WARMUP=0`, `1`, and `3`;
- one BUILD followed by stable REPLAY;
- same-shape/different-value coverage;
- maximum batch, sequence, and cache envelope; and
- the public example completing from the pinned checkpoint in a fresh process.

Record source, dependency, checkpoint, configuration, asset, input, and output
hashes together. A missing operation, unsafe resource plan, or hard semantic
failure blocks the profile. Incomplete parity or task evidence remains
Experimental or Source-only; it cannot be promoted by narrowing the envelope
unless that exact narrower profile passes every applicable gate.

**Done means:** only after every gate passes may the exact profile be added to
the runnable table in [Model support](model_support.md).
