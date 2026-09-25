# Model family implementation map

Use this page to find the public entry point and implementation owner for the
main RhinoForge model families. It is a navigation map, not a support matrix.
Use [Examples](../../docs/model_support.md) to find the relevant configuration
and API. Confirm the exact checkpoint, precision, input envelope, and runtime
assets against that entry point’s admission checks before loading weights.

| Family | Public entry | Implementation owner | Reused execution contracts |
|---|---|---|---|
| Qwen3 and Llama CausalLM | [`RPUModelForCausalLM`](../../docs/api_reference.md#causal-language-models) | [Qwen3 adapter](../../python/rpu_backend/adapters/qwen3.py) and [Llama adapter](../../python/rpu_backend/adapters/llama.py) | [KV cache](../concepts/kv-cache.md), [weight swizzle](../concepts/weight-swizzle.md), and [Graph capture](../concepts/graphcache-capture.md) |
| Qwen3.5 text and Source-only vision profiles | Qwen3.5 adapter plus [`Qwen3_5Cache`](../../docs/api_reference.md#rpucache) | [Qwen3.5 adapter package](../../python/rpu_backend/adapters/qwen3_5/__init__.py) | Separate text/vision admission, profile-specific Graph ownership, and fail-closed numeric-blocked evaluation |
| Qwen3-VL | [`RPUModelForConditionalGeneration`](../../docs/api_reference.md#image-text-models) | [Qwen3-VL adapter package](../../python/rpu_backend/adapters/qwen3_vl/__init__.py) | Vision/text orchestration, [multimodal positions](../concepts/mrope.md), Graph signatures, and logical cache position |
| DINOv3 ViT-B | [`DINOv3Adapter`](../../python/rpu_backend/adapters/dinov3.py) | [DINOv3 adapter](../../python/rpu_backend/adapters/dinov3.py) | Vision planning, patch-embedding ownership, [SPM allocation](../concepts/spm-allocation.md), and Graph capture |
| SigLIP | [`patch_siglip_model_for_rpu_all_layers_once`](../../python/rpu_backend/adapters/siglip.py) | [SigLIP adapter](../../python/rpu_backend/adapters/siglip.py) | Vision-component weight conversion, Graph ownership, and independent output lifetime |
| Pi0.5 | [`Pi05Policy`](../../docs/api_reference.md#policy-apis) | [Pi0.5 policy loader](../../python/rpu_backend/api/policy.py) and [adapter package](../../python/rpu_backend/adapters/pi05/__init__.py) | Multi-component SPM ownership, per-model Graph caches, and profile-specific quantization |
| Wall-OSS | [`WallOssPolicy`](../../docs/api_reference.md#policy-apis) | [Wall-OSS facade](../../python/rpu_backend/api/wall_oss.py) and [adapter package](../../python/rpu_backend/adapters/wall_oss/__init__.py) | Multimodal prefix construction, vision/text/action handoff, and exact-profile execution settings |
| Hy-Embodied | [`HyEmbodiedPolicy`](../../docs/api_reference.md#policy-apis) | [Hy-Embodied facade](../../python/rpu_backend/api/hy_embodied.py) and [adapter package](../../python/rpu_backend/adapters/hy_vla/__init__.py) | [Multi-component handoff](../concepts/component-handoff.md), exact finite-input ownership, fixed three-camera prefix `{192,208,224,240}`, and normalized-action output contract |
| Gemma4 | [`Gemma4Adapter`](../../python/rpu_backend/adapters/gemma4/__init__.py) | [Gemma4 adapter](../../python/rpu_backend/adapters/gemma4/__init__.py) | Mixed decoder geometry, adapter admission, and shared Graph/cache lifecycle |
| GR00T | [Source-only example](../../examples/gr00t.py) | [GR00T adapter package](../../python/rpu_backend/adapters/gr00t/__init__.py) | Exact checkpoint and Cosmos backbone identity, caller-preprocessed input, component handoff, and profile-owned action Graphs |
| LingBot-VLA-V2 | [`Lingbot2Policy.infer(...).actions_normalized`](../../python/rpu_backend/api/lingbot2.py); Source-only | [LingBot2 facade](../../python/rpu_backend/api/lingbot2.py) and [adapter package](../../python/rpu_backend/adapters/lingbot_vla_v2/__init__.py) | Fixed checkpoint and Qwen processor, cold LKN limits, exact graph-arena-pool-v2 preflight, and fail-first rejection of exact `predict_action_chunk()` |
| Galaxea G0.5 | [`patch_g05_policy_for_rpu`](../../python/rpu_backend/adapters/g05/__init__.py); [Source-only integration example](../../examples/g05.py) | [G0.5 adapter package](../../python/rpu_backend/adapters/g05/__init__.py) | Pinned checkpoint/source admission, caller-owned preprocessing, single-frame vision with the controlled numeric-blocked gate, and retained non-commercial licensing; the integration template does not certify full-task quality |
| InternVLA-N1 + NavDP | [InternVLA adapters](../../python/rpu_backend/adapters/internvla_n1/__init__.py) and [`build_navdp`](../../python/rpu_backend/adapters/navdp/runtime.py) | [InternVLA package](../../python/rpu_backend/adapters/internvla_n1/__init__.py) and [NavDP package](../../python/rpu_backend/adapters/navdp/__init__.py) | Caller-pinned assets, standalone component admission, and separate full-policy controlled gates |
| RhinoVLA | [`RhinoVLAPolicy`](../../docs/api_reference.md#policy-apis) | [RhinoVLA facade](../../python/rpu_backend/api/rhinovla.py) and [adapter package](../../python/rpu_backend/adapters/rhinovla/__init__.py) | Qwen3-VL prefix processing plus an action-expert path; the separate model repository owns checkpoint composition and preprocessing |

## Family-specific runtime contracts

Shared fused-decoder RMSNorm scheduling is resolved from tensor shape and the
immutable operator payloads available in the loaded runtime set. Model handles
declare a capability ceiling; they do not select a route by family name or
checkpoint size. The same resolver is used by Gemma/Pi0.5, Qwen, and Hy-VLA
call sites, while exact profiles may request a narrower capability and fail
closed when its payload is unavailable.
[RMSNorm resolver](../../src/core/rpu_kernel_decls.h)
[RMSNorm launcher](../../src/ops/rpu_rmsnorm.cpp)

The tensor allocator is likewise one cold process policy. An adapter that needs
cached storage must claim the shared policy before its first RPU tensor
allocation; another live profile cannot switch it later. Use a fresh process
to change allocator mode.
[Runtime configuration](../../docs/runtime_config.md#memory-and-coherency-controls)

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
  `Qwen3_5Adapter` with `Qwen3_5Cache`. The 2B/4B Vision profiles are
  Source-only/numeric-blocked because official real-image gates fail; image
  inference is rejected by default, and exact
  `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1` admits controlled evaluation only.
  Video remains outside the public scope. Do not infer image support from a
  successful text run.
  [Examples](../../docs/model_support.md)
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
- The adapter claims the shared caching policy before its first RPU or reserved
  Launch allocation. Long-running varied-image validation must prove its
  address-free mapping count stays bounded.
- The legacy 32B W8A16 path remains Source-only and controlled/uncertified:
  only Text7 is INT8; the head, embedding and Vision stay FP16. Its independent
  TP8 planning bounds (336 physical prefill rows, chunk at most 64) do not
  certify that entire length range or set KV capacity. Validate the public CLI,
  same-teacher CPU numerical comparisons and Graph lifetime for the actual
  input envelope. Full-range, task-quality and performance certification remain
  separate gates; the explicit opt-in remains required. Runtime-quantized 32B
  is a distinct recipe and cannot supply this legacy profile's evidence.
  [Qwen3-VL configurations](../../examples/configs/qwen3_vl/README.md)

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
- The adapter claims the common caching allocator policy only after its pure
  host profile checks and before the first RPU tensor allocation. This removes
  HostDDR allocation churn without creating a Pi-specific backend allocator or
  operator schedule.
- The input tensor dictionary must come from checkpoint-compatible
  preprocessing. Camera keys, image geometry, token length, and auxiliary
  fields belong to that checkpoint contract.
  [Pi0.5 example](../../examples/pi05.py)
- For fixed-input comparisons, pass the same explicit CPU `float32` noise to
  graph preparation and inference, or use `input.noise` in the example TOML.
  Shape and finite-value validation precede inference; omitting noise keeps
  the default sampling behavior. Keep numerical safety checks enabled.
  [Policy API](../../docs/api_reference.md#policy-apis)
  [Noise validation tests](../../tests/test_pi05_explicit_noise.py)
- Cold installation reserves capacity for four retained component/safety
  Graphs and one immediate queue before weight transformation. The example
  constructs persistent tensors outside inference mode so version-based
  prefix caching remains available; the request still uses the configured
  inference mode.
  [Pi0.5 installation](../../python/rpu_backend/adapters/pi05/__init__.py)
  [Example runner](../../examples/_runner.py)

### Wall-OSS

- Construct the policy through `WallOssPolicy.from_checkpoint(...)` or
  `from_pretrained(...)`, bind one complete profile, move it to RPU once, and
  prepare only the finite graph envelope owned by that policy. Directly mixing
  individual runtime switches from other VLA profiles is not a supported
  configuration.
  [Policy API](../../docs/api_reference.md#policy-apis)
- `WallOssPolicy` is the only public full-model entry. The public adapter
  package exposes component paths only; its full-model orchestrator is private.
- Images, instruction, proprioception, masks, precision, prefix envelope, and
  action settings are part of the request/profile contract. The facade
  validates them before execution; a source-level adapter path is not a bypass
  for those checks.
  [Wall-OSS facade](../../python/rpu_backend/api/wall_oss.py)
- The pinned public prompt follows `wall-x` with
  `use_embodied_system_prompt_ratio=0.0`: the normal path uses the exact plain
  `You are a helpful assistant.` system message. Diagnostic prompt selectors
  define different inputs and cannot inherit public-profile evidence.
- The Source-only public image envelope is exactly two `[1,32,32]` grids.
  Packed Vision stays disabled: each image uses the shared `768+256` KV-first
  plan sequentially, preserving image boundaries and merger row order.
- Text prefill uses the shared bounded planner. Its configured chunk cap is a
  ceiling, not an equal-partition requirement; kernel validity and the final
  SPM dry resolver choose every physical multi-chunk plan.
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

1. Find the configuration and API through [Examples](../../docs/model_support.md).
   Keep their exact input, precision, and runtime restrictions. Source-only and
   controlled evaluation paths do not establish model certification.
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
