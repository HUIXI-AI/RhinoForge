# Model family implementation map

Use this page to find the public entry point and implementation owner for the
main RhinoForge model families. It is a navigation map, not a support matrix.
Always confirm the exact checkpoint, precision, input envelope, and runtime
assets in [Model support](../../docs/model_support.md) before loading weights.

| Family | Public entry | Implementation owner | Reused execution contracts |
|---|---|---|---|
| Qwen3 and Llama CausalLM | [`RPUModelForCausalLM`](../../docs/api_reference.md#causal-language-models) | [Qwen3 adapter](../../python/rpu_backend/adapters/qwen3.py) and [Llama adapter](../../python/rpu_backend/adapters/llama.py) | [KV cache](../concepts/kv-cache.md), [weight swizzle](../concepts/weight-swizzle.md), and [Graph capture](../concepts/graphcache-capture.md) |
| Qwen3.5 text and controlled vision profiles | Qwen3.5 adapter plus [`Qwen3_5Cache`](../../docs/api_reference.md#rpucache) | [Qwen3.5 adapter package](../../python/rpu_backend/adapters/qwen3_5/__init__.py) | Separate text/vision admission, profile-specific Graph ownership, and fail-closed evaluation gates |
| Qwen3-VL | [`RPUModelForConditionalGeneration`](../../docs/api_reference.md#image-text-models) | [Qwen3-VL adapter package](../../python/rpu_backend/adapters/qwen3_vl/__init__.py) | Vision/text orchestration, [multimodal positions](../concepts/mrope.md), Graph signatures, and logical cache position |
| Pi0.5 | [`Pi05Policy`](../../docs/api_reference.md#policy-apis) | [Pi0.5 policy loader](../../python/rpu_backend/api/policy.py) and [adapter package](../../python/rpu_backend/adapters/pi05/__init__.py) | Multi-component SPM ownership, per-model Graph caches, and profile-specific quantization |
| Wall-OSS | [`WallOssPolicy`](../../docs/api_reference.md#policy-apis) | [Wall-OSS facade](../../python/rpu_backend/api/wall_oss.py) and [adapter package](../../python/rpu_backend/adapters/wall_oss/__init__.py) | Multimodal prefix construction, vision/text/action handoff, and exact-profile execution settings |
| RhinoVLA | [`RhinoVLAPolicy`](../../docs/api_reference.md#policy-apis) | [RhinoVLA facade](../../python/rpu_backend/api/rhinovla.py) and [adapter package](../../python/rpu_backend/adapters/rhinovla/__init__.py) | Qwen3-VL prefix processing plus an action-expert path; the separate model repository owns checkpoint composition and preprocessing |

## Family-specific runtime contracts

### Qwen3

- The dense causal path uses the shared fused-decoder framework, keeps
  cross-layer intermediates in SPM, and owns generation state through
  `RPUCache`. Use the public loader and cache factory instead of constructing a
  native handle or cache layout directly.
  [CausalLM API](../../docs/api_reference.md#causal-language-models)
- `rpu_execution.prefill` is cold, per-handle configuration. Finalize it before
  RPU installation, create the cache after the model profile is fixed, and
  size the cache for prompt plus decode.
  [Qwen3 example](../../examples/causal_lm.py)
- Weight conversion is irreversible and exactly once. If construction fails
  after conversion starts, reload a clean model instance rather than retrying
  the partially transformed object.
  [Weight swizzle](../concepts/weight-swizzle.md)

### Qwen3.5

- Text and vision are separate support scopes. Dense text profiles use
  `Qwen3_5Adapter` with `Qwen3_5Cache`; the named vision profiles remain behind
  a fail-closed evaluation gate. Do not infer image or video support from a
  successful text run.
  [Model support](../../docs/model_support.md)
- Variable-length text prefill uses a bounded one-shot Graph contract, while
  decode uses retained GraphCache replay. Verification must apply the lifecycle
  gate for the stage being tested rather than requiring one Graph policy for
  both stages.
  [Graph capture](../concepts/graphcache-capture.md)
- Position, cache, image geometry, and vision-to-text handoff values are
  semantic inputs. Keep them in the Graph signature or update them through the
  documented mutable-data path; same shape does not imply same meaning.
  [Qwen3.5 adapter](../../python/rpu_backend/adapters/qwen3_5/__init__.py)

### Qwen3-VL

- The adapter composes an image tower, multimodal position handling, visual
  feature injection, and the fused text decoder. Processor-produced image
  geometry and position data are semantic inputs; do not replace them with
  copied constants from another checkpoint or image shape.
  [Qwen3-VL adapter](../../python/rpu_backend/adapters/qwen3_vl/__init__.py)
- Initial-prefill execution padding is adapter-owned. Physical execution rows,
  multimodal positions, visual features, returned sequence length, and logical
  KV-cache position must remain consistent.
  [Multimodal positions](../concepts/mrope.md)
- Build `RPUCache` from the language model, retain independent outputs when
  processing several images, and prove the Graph lifecycle for every admitted
  geometry/signature.
  [Qwen3-VL example](../../examples/qwen3_vl.py)

### Pi0.5

- The policy composes visual encoding, a vision-language model, and a
  flow-matching action path. The component adapters own dedicated GraphCache
  instances and temporary SPM lifetimes; component handoff is an ownership
  boundary, not a reason to use a process-global Graph cache.
  [Pi0.5 adapter](../../python/rpu_backend/adapters/pi05/__init__.py)
- Construct through `Pi05Policy`, move the policy to RPU once, optionally call
  `prepare_graphs`, and use `predict_action_chunk` or `select_action`. Configure
  prefill, vision, and action planning through public `rpu_execution`, not
  undocumented module attributes or process-global chunk settings.
  [Policy API](../../docs/api_reference.md#policy-apis)
- The input tensor dictionary must come from checkpoint-compatible
  preprocessing. Camera keys, image geometry, token length, and auxiliary
  fields belong to that checkpoint contract.
  [Pi0.5 example](../../examples/pi05.py)

### Wall-OSS

- Construct the policy through `WallOssPolicy.from_checkpoint(...)` or
  `from_pretrained(...)`, bind one complete profile, move it to RPU once, and
  prepare only the finite graph envelope owned by that policy. Directly mixing
  individual runtime switches from other VLA profiles is not a supported
  configuration.
  [Policy API](../../docs/api_reference.md#policy-apis)
- Images, instruction, proprioception, masks, precision, prefix envelope, and
  action settings are part of the request/profile contract. The facade
  validates them before execution; a source-level adapter path is not a bypass
  for those checks.
  [Wall-OSS facade](../../python/rpu_backend/api/wall_oss.py)
- Vision, prefix/text, cache, and action stages retain independent ownership
  and correctness evidence. When changing batching, replay, fusion, or
  quantization, treat the result as a new runtime profile until semantic and
  task-level equivalence are demonstrated.
  [Runtime profiles](../concepts/runtime-profiles.md)

### RhinoVLA

- The backend facade wraps a model-repository-owned runtime that combines a
  Qwen3-VL-style visual/text prefix with an action-expert path. The external
  runtime factory remains responsible for checkpoint composition,
  preprocessing, irreversible RPU conversion, and live-input validation.
  [RhinoVLA facade](../../python/rpu_backend/api/rhinovla.py)
- Use only an installed, reviewed `runtime_factory` callable or
  `module:callable` extension. When `rpu_execution` is supplied, the factory and
  returned runtime must declare their capabilities, expose the effective
  configuration, and publish a fresh resolved plan; the facade rejects silent
  disagreement.
  [Policy API](../../docs/api_reference.md#policy-apis)
- `prepare_graphs` warms the finite runtime profile and `predict` or
  `predict_action_chunk` executes one request. Validate visual, prefix/text,
  cache, and action boundaries separately; a final action result alone cannot
  localize an upstream component error.
  [RhinoVLA example](../../examples/rhinovla.py)

## How to use the map

1. Check the row in [Model support](../../docs/model_support.md). `Limited`,
   `Experimental`, `Component-only`, and `Source-only` are distinct contracts.
2. Start from the public API and its checked example in
   [Model execution and profiling](../../docs/model_testing.md).
3. Read the owning adapter before searching native source. The adapter owns
   preflight, weight transformation, cache construction, Graph signatures, and
   teardown.
4. Reuse the indexed [SPM](../concepts/spm-allocation.md),
   [DMA](../concepts/ddr-dma-wrappers.md),
   [multi-core](../concepts/multicore-broadcast.md), and
   [attention-layout](../concepts/sdpa-layout.md) contracts instead of creating
   a model-local alternative.
5. Use the [porting playbook](porting-playbook.md) when the target does not fit
   an existing exact profile.

Source presence or a matching architecture name never promotes a profile. A
missing checkpoint, restricted runtime asset, numerical gate, or end-to-end
input contract keeps the profile at its documented status.
