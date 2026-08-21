# Multimodal rotary positions

Multimodal rotary position encoding assigns each token three logical
coordinates, conventionally temporal, height, and width, instead of one scalar
position. RhinoForge uses this contract in Qwen3-VL and in model-specific
profiles that reuse its multimodal text path. Profile support still comes from
[Model support](../../docs/model_support.md), not from the presence of this
helper.

## Public boundary

- The host precomputation helper accepts non-negative integer
  `position_ids` with shape `[sequence, 3]`.
  [Position-table helper](../../python/rpu_backend/runtime/rope_partial.py)
- `mrope_section` contains exactly three non-negative integer section sizes.
  Their sum must equal half of the configured rotary dimension. Read it from
  the model configuration; do not hardcode another model's values.
  [Qwen3-VL text installation](../../python/rpu_backend/adapters/qwen3_vl/text.py)
- The Qwen3-VL causal-prefill boundary accepts the Hugging Face form
  `[3, batch, sequence]` and the token-major form `[sequence, 3]`. Preserve the
  axis order when adapting another model.
  [Qwen3-VL adapter](../../python/rpu_backend/adapters/qwen3_vl/__init__.py)
- Padding rows need valid position rows, but they must not change the logical
  output length or KV-cache position. The adapter pads positions consistently,
  slices the returned output, and preserves logical cache ownership.
  [Architecture](../../docs/architecture.md#spm-design)
- Position coordinates are semantic inputs. A retained Graph must either key
  on every result-changing position property or update the live position data
  through its reviewed mutable-data contract.
  [GraphCache capture](graphcache-capture.md)

When all three coordinates for a token are equal, each rotary section uses the
same logical position. This is useful for checking text-only suffixes, but it
does not justify replacing a genuinely multimodal position table with a 1-D
one.

## Sources

- [Public position-table helpers](../../python/rpu_backend/runtime/rope_partial.py)
- [Qwen3-VL text adapter](../../python/rpu_backend/adapters/qwen3_vl/text.py)
- [Qwen3-VL top-level adapter](../../python/rpu_backend/adapters/qwen3_vl/__init__.py)
- [Model porting: graph and cache invariants](../../docs/model_porting.md#7-graph-and-cache-invariants)
