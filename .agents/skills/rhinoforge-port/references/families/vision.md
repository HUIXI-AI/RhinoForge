# Vision encoder playbook

Use this after an accepted assessment classifies the target as a vision
encoder. Follow the shared implementation and evidence rules in
[model porting](../../../../../docs/model_porting.md).

Public references are the [SigLIP adapter](../../../../../python/rpu_backend/adapters/siglip.py),
the [DINOv3 adapter](../../../../../python/rpu_backend/adapters/dinov3.py), and
the public [SPM lifetime contract](../../../../../knowledge/concepts/spm-allocation.md).

1. Trace image preprocessing, patch or image embedding, position handling,
   every encoder block, pooling, and returned features on CPU.
2. Decide who owns patch embedding and any projection. Do not apply the same
   projection in both Python and the fused path. If patch embedding remains on
   the host, report it as explicit CPU glue rather than RPU execution.
3. Match bidirectional attention, masks, normalization, activation, residual
   ordering, and position semantics exactly.
4. Define a deterministic SPM manifest for the maximum admitted image and
   token envelope. Keep replay-persistent state explicit.
5. Preserve the dtype and values of loaded position data. Prefer the exact
   upstream position helper over a nearby approximation.
6. Include geometry and every semantic table in the Graph signature, or update
   live values through the documented mutable path.
7. Use SPM aliases only after proving producer and consumer lifetimes do not
   overlap; validate a shape that crosses each affected chunk boundary.
8. Give returned features independent storage when the caller may retain
   multiple images or forwards.
9. Verify the component and task evidence required by the public
   [validation policy](../../../../../docs/validation_policy.md), repeated
   BUILD-to-REPLAY behavior, same-shape/different-geometry inputs, multiple
   images, and the maximum admitted envelope.

Do not infer support for another image size, precision, or pooling contract
from one passing encoder shape.
