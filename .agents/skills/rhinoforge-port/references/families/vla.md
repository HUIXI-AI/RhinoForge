# Vision-language-action policy playbook

Use this after an accepted assessment classifies the target as a VLA. Follow
the shared implementation and evidence rules in
[model porting](../../../../../docs/model_porting.md).

Public composed references are the
[Pi0.5 adapter](../../../../../python/rpu_backend/adapters/pi05/__init__.py) and
[Wall-OSS adapter](../../../../../python/rpu_backend/adapters/wall_oss/__init__.py).

1. Start with the VLM component-ownership map, then add state and action
   preprocessing, action-expert inputs, iterative update state, scheduler or
   time inputs, and output normalization.
2. Trace one complete CPU policy call through the final action. Pin the number
   of update steps, action horizon and dimension, camera/input schema, random
   input or noise, and returned policy state.
3. Reuse a public VLM or action subsystem only when its checkpoint schema,
   normalization, position math, and update rule match exactly.
4. Define ownership at every subsystem boundary. Reset temporary SPM and
   release the current compute owner before handing execution to the next
   subsystem.
5. Give each policy component an explicit model-owned Graph cache. Do not use a
   process-global cache to bridge subsystem ownership.
6. If iterative steps reuse a computed prefix, preserve that prefix and rewind
   only the logical suffix through the public cache API before each step.
7. Keep iteration state and caller-retained outputs at valid addresses across
   replay. Use mutable DMA for live caller inputs or fresh outputs.
8. If the model keeps its iterative update on the host, preserve the upstream
   precision and formula exactly, report the CPU boundary, and verify every
   step as well as the final action.
9. Include all values that change action output in the Graph signature or a
   validated mutable update path.
10. Verify intermediate component boundaries, every update step, and the final
   action under the public
   [validation policy](../../../../../docs/validation_policy.md), plus warmup
   variants, multi-camera or multi-input behavior, Graph lifecycle, and the
   maximum admitted policy envelope.

Numerical execution is not physical-system or robot-readiness certification.
Report only the exact inference profile and gates that were actually run.
