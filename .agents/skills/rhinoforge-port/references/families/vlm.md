# Vision-language model playbook

Use this after an accepted assessment classifies the target as a VLM. Follow
the shared implementation and evidence rules in
[model porting](../../../../../docs/model_porting.md).

Use the [Qwen3-VL adapter](../../../../../python/rpu_backend/adapters/qwen3_vl/__init__.py)
as the public composed reference and
[multimodal positions](../../../../../knowledge/concepts/mrope.md) for position,
padding, and Graph contracts.

1. Draw a component-ownership map for preprocessing, vision tower, projector
   or merger, image/text composition, decoder, KV cache, and output head.
2. Trace the CPU reference across every component boundary. Record shapes,
   dtypes, masks, position data, placeholder counts, and returned outputs.
3. Reuse a public vision or decoder path only when its exact math and profile
   match. An architecture name does not establish compatibility.
4. Apply each projection, layout transformation, and weight conversion exactly
   once. If the candidate injects vision features into intermediate decoder
   layers, map that ownership explicitly instead of inferring it from a related
   architecture.
5. Validate placeholder and image-feature counts before composing the text
   stream. If composition remains on the host, report that explicit CPU glue
   boundary and verify it end to end.
6. Include image geometry, position data, masks, and other semantic values in
   Graph signatures or update them through a reviewed mutable transfer.
7. When an execution path pads initial prefill, keep physical rows, position
   data, visual features, returned length, and logical cache position
   consistent; slice outputs and rewind cache state to the logical length.
8. Give each caller-retained output independent storage across images and
   forwards.
9. Verify raw vision output, post-projector or post-merger features, final text
   output, prefill and decode under the public
   [validation policy](../../../../../docs/validation_policy.md), plus multiple
   images, warmup stability, and the applicable Graph lifecycle against the
   same immutable profile.

Do not promote a VLM from component-level passes alone; the composed public
end-to-end entry point is the final functional gate.
