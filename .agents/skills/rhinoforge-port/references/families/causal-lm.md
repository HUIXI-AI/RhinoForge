# Decoder CausalLM playbook

## Route

Use this reference after an accepted assessment identifies a decoder-only
CausalLM. An Existing path needs no port. A Blocked or uncertified assessment
stops before code changes. An Adapter-only result follows the public
[step-by-step checklist](../../../../../docs/new_model_step_by_step.md);
runtime extensions and fused subsystems follow
[model porting](../../../../../docs/model_porting.md).

## Must-check capabilities

- attention: causal mask, Q/K normalization, head and KV-head geometry, and
  returned layout;
- cache: prefill, teacher-forced decode, capacity, logical position, and reset;
- position: exact rotary or other position convention and padding semantics;
- norm and MLP: type, epsilon, placement, residual ordering, activation, bias,
  and projection ownership;
- linear/weights: tied head behavior, specialized projections, quantization
  metadata, and exactly-once transformation;
- logits/generation: output head, cache return, generation path, and public
  return type; and
- runtime/Graph: exact-profile preflight, cold execution settings, SPM/DMA
  ownership, complete signatures, independent outputs, and teardown.

## Public reference adapters

- [Qwen3](../../../../../python/rpu_backend/adapters/qwen3.py) for the shared
  Q/K-normalized decoder path.
- [Llama](../../../../../python/rpu_backend/adapters/llama.py) for the shared
  decoder path without Q/K normalization.
- [Minimal template](../../../../../python/rpu_backend/adapters/_template/minimal_causal_lm.py)
  for an Adapter-only port; use the adjacent annotated `adapter.py` only when
  the extra structure is required.

An architecture name does not prove mathematical compatibility with either
decoder path.

## Stop criteria

- unsupported profiles fail before weight loading or irreversible conversion;
- the exact profile's reviewed same-dtype, FP32-anchor, and task contract passes
  under the public [validation policy](../../../../../docs/validation_policy.md)
  for complete prefill and teacher-forced decode;
- warmup variants are stable and a repeated signature proves the applicable
  Graph lifecycle without cache growth;
- same-shape semantic changes, multiple retained outputs, and the maximum
  admitted sequence/cache envelope pass;
- board-free registry/configuration/build checks pass; and
- the public TOML entry completes in a clean process with the pinned model,
  runtime library, and operator asset.
