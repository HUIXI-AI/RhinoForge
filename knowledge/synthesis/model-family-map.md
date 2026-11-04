# Model family implementation map

Use this page to find the public entry point and implementation owner for the
main RhinoForge model families. Select the checkpoint, precision and input configuration from [examples](../../docs/model_support.md) before loading weights.

| Family | Public entry | Implementation owner | Reused execution contracts |
|---|---|---|---|
| Qwen3 and Llama CausalLM | [`RPUModelForCausalLM`](../../docs/api_reference.md#causal-language-models) | [Qwen3 adapter](../../python/rpu_backend/adapters/qwen3.py) and [Llama adapter](../../python/rpu_backend/adapters/llama.py) | [KV cache](../concepts/kv-cache.md), [weight swizzle](../concepts/weight-swizzle.md), and [Graph capture](../concepts/graphcache-capture.md) |
| Qwen3.5 dense and MoE | [`RPUModelForConditionalGeneration`](../../docs/api_reference.md#image-text-models), `Qwen3_5Cache` / `Qwen3_5MoeCache` | [Dense adapter](../../python/rpu_backend/adapters/qwen3_5/__init__.py), [MoE adapter](../../python/rpu_backend/adapters/qwen3_5_moe/__init__.py) | Separate text/vision and checkpoint admission, profile-specific Graph and recurrent-state ownership |
| Qwen3-VL | [`RPUModelForConditionalGeneration`](../../docs/api_reference.md#image-text-models) | [Qwen3-VL adapter package](../../python/rpu_backend/adapters/qwen3_vl/__init__.py) | Vision/text orchestration, [multimodal positions](../concepts/mrope.md), Graph signatures, and logical cache position |
| DINOv3 ViT-B | [`DINOv3Adapter`](../../python/rpu_backend/adapters/dinov3.py) | [DINOv3 adapter](../../python/rpu_backend/adapters/dinov3.py) | Vision planning, patch-embedding ownership, [SPM allocation](../concepts/spm-allocation.md), and Graph capture |
| SigLIP | [`patch_siglip_model_for_rpu_all_layers_once`](../../python/rpu_backend/adapters/siglip.py) | [SigLIP adapter](../../python/rpu_backend/adapters/siglip.py) | Vision-component weight conversion, Graph ownership, and independent output lifetime |
| Pi0.5 | [`Pi05Policy`](../../docs/api_reference.md#policy-apis) | [Pi0.5 policy loader](../../python/rpu_backend/api/policy.py) and [adapter package](../../python/rpu_backend/adapters/pi05/__init__.py) | Multi-component SPM ownership, per-model Graph caches, and profile-specific quantization |
| Wall-OSS | [`WallOssPolicy`](../../docs/api_reference.md#policy-apis) | [Wall-OSS facade](../../python/rpu_backend/api/wall_oss.py) and [adapter package](../../python/rpu_backend/adapters/wall_oss/__init__.py) | Multimodal prefix construction, vision/text/action handoff, and exact-profile execution settings |
| Hy-Embodied | [`HyEmbodiedPolicy`](../../docs/api_reference.md#policy-apis) | [Hy-Embodied facade](../../python/rpu_backend/api/hy_embodied.py) and [adapter package](../../python/rpu_backend/adapters/hy_vla/__init__.py) | [Multi-component handoff](../concepts/component-handoff.md), exact finite-input ownership, fixed three-camera prefix `{192,208,224,240}`, and normalized-action output contract |
| Gemma4 | [`Gemma4Adapter`](../../python/rpu_backend/adapters/gemma4/__init__.py) | [Gemma4 adapter](../../python/rpu_backend/adapters/gemma4/__init__.py) | Mixed decoder geometry, adapter admission, and shared Graph/cache lifecycle |
| GR00T | [Source-only example](../../examples/run.py) | [GR00T adapter package](../../python/rpu_backend/adapters/gr00t/__init__.py) | Exact checkpoint and Cosmos backbone identity, caller-preprocessed input, component handoff, and profile-owned action Graphs |
| LingBot-VLA-V2 | [`build_lingbot_vla_v2` / `get_action`](../../examples/configs/lingbot/v2/README.md) | [Runtime builder](../../python/rpu_backend/adapters/lingbot_vla_v2/runtime.py); [legacy facade](../../python/rpu_backend/api/lingbot2.py) | Training YAML, FP16/W8A16, two/three cameras, sparse Top4 and retained component Graphs |
| Galaxea G0.5 | [`patch_g05_policy_for_rpu`](../../python/rpu_backend/adapters/g05/__init__.py); [Source-only integration example](../../examples/run.py) | [G0.5 adapter package](../../python/rpu_backend/adapters/g05/__init__.py) | Pinned checkpoint/source admission, caller-owned preprocessing, single-frame vision with the controlled numeric-blocked gate, and retained non-commercial licensing |
| InternVLA-N1 + NavDP | [InternVLA adapters](../../python/rpu_backend/adapters/internvla_n1/__init__.py) and [`build_navdp`](../../python/rpu_backend/adapters/navdp/runtime.py) | [InternVLA package](../../python/rpu_backend/adapters/internvla_n1/__init__.py) and [NavDP package](../../python/rpu_backend/adapters/navdp/__init__.py) | Caller-pinned assets, standalone component admission, and separate full-policy controlled gates |
| RhinoVLA | [`RhinoVLAPolicy`](../../docs/api_reference.md#policy-apis) | [RhinoVLA facade](../../python/rpu_backend/api/rhinovla.py) and [adapter package](../../python/rpu_backend/adapters/rhinovla/__init__.py) | Qwen3-VL prefix processing plus an action-expert path; the separate model repository owns checkpoint composition and preprocessing |

## Family-specific runtime contracts

Shared fused-decoder RMSNorm scheduling is resolved from tensor shape and the
immutable operator payloads available in the loaded runtime set. Model handles
declare a capability ceiling; they do not select a route by family name or
checkpoint size. The same resolver is used by Gemma/Pi0.5, Qwen, and Hy-VLA
call sites, while exact profiles may request a narrower capability and fail
closed when its payload is unavailable. [RMSNorm resolver](../../src/core/rpu_kernel_decls.h)
[RMSNorm launcher](../../src/ops/rpu_rmsnorm.cpp)

The tensor allocator is likewise one cold process policy. An adapter that needs
cached storage must claim the shared policy before its first device tensor
allocation; another live profile cannot switch it later. Use a fresh process
to change allocator mode. [Runtime configuration](../../docs/runtime_config.md#memory-and-coherency-controls)

### Qwen3

- The dense causal path uses the shared fused-decoder framework, keeps
  cross-layer intermediates in SPM, and owns generation state through
  `RPUCache`. Use the public loader and cache factory instead of constructing a
  native handle or cache layout directly. [CausalLM API](../../docs/api_reference.md#causal-language-models)
- Set `rpu_execution` before device installation and size the model's cache for
  prompt plus decode. Core count and precision are cold; supported chunk/padding
  changes use the execution API between requests. [Execution API](../../docs/api_reference.md#execution-configuration-and-updates)
- Weight conversion is irreversible and exactly once. If construction fails
  after conversion starts, reload a clean model instance rather than retrying
  the partially transformed object. [Weight swizzle](../concepts/weight-swizzle.md)

### Qwen3.5

- Text uses `Qwen3_5Adapter` and `Qwen3_5Cache`. Image inputs require `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1`; quantized inputs also require `QWEN3_5_QUANT_ALLOW_UNCERTIFIED=1`. Video is unsupported.
- Dense 0.8B/2B/4B text retains one prefill entry. Exact TP8 9B pure text also retains prefill for FP16 and uniform W8/W4 group32. Other 9B and both legacy/official 27B use bounded one-shot prefill; decode retains a separate Graph.
- Official 27B uses caching allocation, four cold Graph arenas and finalized uploads from CPU weights. W8 supports root4/6/8, W4 group32 root8. Root6 assigns MLP6 with attention/GDN/head/Vision4. [Configuration guide](../../examples/configs/qwen3_5/README.md).
- Position, cache, image geometry and vision-to-text handoff are semantic inputs. Include them in Graph signatures or refresh them through the mutable-data path. [Adapter](../../python/rpu_backend/adapters/qwen3_5/__init__.py).

### Qwen3-VL

- The adapter composes an image tower, multimodal position handling, visual
  feature injection, and the fused text decoder. Processor-produced image
  geometry and position data are semantic inputs; do not replace them with
  copied constants from another checkpoint or image shape. [Qwen3-VL adapter](../../python/rpu_backend/adapters/qwen3_vl/__init__.py)
- Initial-prefill execution padding is adapter-owned. Physical execution rows,
  multimodal positions, visual features, returned sequence length, and logical
  KV-cache position must remain consistent. [Multimodal positions](../concepts/mrope.md)
- Build `RPUCache` from the language model, retain independent outputs when
  processing several images, and prove the Graph lifecycle for every admitted
  geometry/signature. [Qwen3-VL example](../../examples/run.py)
- The adapter claims the shared caching policy before its first device or reserved
  Launch allocation. Long-running varied-image validation must prove its
  address-free mapping count stays bounded.
- Legacy 32B Text-W8 is a separate opt-in profile: Text7 is INT8; head, embedding and Vision remain FP16. Planning caps are 336 physical prefill rows and chunk 64; cache capacity and actual request admission are checked separately. [Configurations](../../examples/configs/qwen3_vl/README.md).

### Pi0.5

- The policy composes visual encoding, a vision-language model, and a
  flow-matching action path. The component adapters own dedicated GraphCache
  instances and temporary SPM lifetimes; component handoff is an ownership
  boundary, not a reason to use a process-global Graph cache. [Pi0.5 adapter](../../python/rpu_backend/adapters/pi05/__init__.py)
- Construct through `Pi05Policy`, move the policy to device once, optionally call
  `prepare_graphs`, and use `predict_action_chunk` or `select_action`. Configure
  prefill, vision, and action planning through public `rpu_execution`, not
  undocumented module attributes or process-global chunk settings. [Policy API](../../docs/api_reference.md#policy-apis)
- The adapter claims the common caching allocator policy only after its pure
  host profile checks and before the first device tensor allocation. This removes
  HostDDR allocation churn without creating a Pi-specific backend allocator or
  operator schedule.
- The input tensor dictionary must come from checkpoint-compatible
  preprocessing. Camera keys, image geometry, token length, and auxiliary
  fields belong to that checkpoint contract. [Pi0.5 example](../../examples/run.py)
- For fixed-input comparisons, pass the same explicit CPU `float32` noise to
  graph preparation and inference, or use `input.noise` in the example TOML.
  Shape and finite-value validation precede inference; omitting noise keeps
  the default sampling behavior. Keep numerical safety checks enabled. [Policy API](../../docs/api_reference.md#policy-apis)
  [Noise validation tests](../../tests/test_pi05_explicit_noise.py)
- Cold installation reserves capacity for four retained component/safety
  Graphs and one immediate queue before weight transformation. The example
  constructs persistent tensors outside inference mode so version-based
  prefix caching remains available; the request still uses the configured
  inference mode. [Pi0.5 installation](../../python/rpu_backend/adapters/pi05/__init__.py)
  [Example runner](../../examples/common/measurement.py)

### Wall-OSS

- Construct the policy through `WallOssPolicy.from_checkpoint(...)` or
  `from_pretrained(...)`, bind one complete profile, move it to device once, and
  prepare only the finite graph envelope owned by that policy. Directly mixing
  individual runtime switches from other VLA profiles is not a supported
  configuration. [Policy API](../../docs/api_reference.md#policy-apis)
- Applications can use `WallOssPolicy`; direct integrations can use the exported
  `build_wall_oss_vla` builder and component builders.
- Images, instruction, proprioception, masks, precision, prefix envelope, and
  action settings are part of the request/profile contract. The facade
  validates them before execution; a source-level adapter path is not a bypass
  for those checks. [Wall-OSS facade](../../python/rpu_backend/api/wall_oss.py)
- The pinned public prompt follows `wall-x` with
  `use_embodied_system_prompt_ratio=0.0`: the normal path uses the exact plain
  `You are a helpful assistant.` system message. Diagnostic prompt selectors
  define different inputs and cannot inherit public-profile evidence.
- Camera names and action geometry bind at construction. The request supplies
  one image per camera; compatible grids may use packed Vision within the
  configured capacity.
- Text prefill uses the shared bounded planner. An integer `chunk_size` is an
  exact request; native shape and SPM checks must admit the complete plan.
- Vision, prefix/text, cache, and action stages retain independent ownership
  and correctness evidence. When changing batching, replay, fusion, or
  quantization, treat the result as a new runtime profile until semantic and
  task-level equivalence are demonstrated. [Runtime profiles](../concepts/runtime-profiles.md)

### RhinoVLA

- The backend facade wraps a model-repository-owned runtime that combines a
  Qwen3-VL-style visual/text prefix with an action-expert path. The external
  runtime factory remains responsible for checkpoint composition,
  preprocessing, irreversible device conversion, and live-input validation. [RhinoVLA facade](../../python/rpu_backend/api/rhinovla.py)
- Use only an installed, reviewed `runtime_factory` callable or
  `module:callable` extension. When `rpu_execution` is supplied, the factory and
  returned runtime must declare their capabilities, expose the effective
  configuration, and publish a fresh resolved plan; the facade rejects silent
  disagreement. [Policy API](../../docs/api_reference.md#policy-apis)
- `prepare_graphs` warms the finite runtime profile and `predict` or
  `predict_action_chunk` executes one request. Validate visual, prefix/text,
  cache, and action boundaries separately; a final action result alone cannot
  localize an upstream component error. [RhinoVLA example](../../examples/run.py)

- The pinned built-in v3 factory has a separate cold `rtc=True` contract for
  the existing FP16 and expert/full-expert/full-pipeline W8A16 recipes with
  30×96/18-layer/ten-step inference. Preserve each recipe's fusion, resident
  AdaRMS, HIGH, vector and partial-RoPE settings; RTC does not change precision.
  The five factory recipes have separate RTC template identities, while the
  nine controlled recipes use `run --rtc --delay 0..6` (one integer per run).
  Requests pair a CPU floating
  `action_prefix` with Python-integer `delay=0..6`; validate them before loading
  model assets. Every step uses zero time for committed actions and the mean
  action-time embedding for state, then preserves the masked prefix. Do not
  treat a final output overwrite or a passing non-RTC case as RTC evidence.
  Delay belongs to the Graph signature; prefix values must refresh for replay.
  Construction reserves ten arenas for seven action variants, retained
  vision/text graphs and one transient preparation graph, without a user
  environment override. Validation uses three equal-sized views and P230; keep
  template qin token/mask/grid contents and pixel geometry cold-bound. Rebuild
  the runtime when they change. Unequal view sizes are outside this validation,
  and forced oneshot is rejected. Keep Python and native RTC builds matched when updating an older
  environment bundle. Do not extend this contract to v1 or unimplemented
  precision formats, and do not inherit non-RTC hardware results for RTC.
  [RTC contract and verification status](../../examples/configs/rhinovla/v3/README.md#rtc),
  [request validation](../../python/rpu_backend/adapters/rhinovla/rtc.py)

## How to use the map

1. Find the configuration and API through [Examples](../../docs/model_support.md).
   Keep their exact input, precision, and runtime restrictions. Source-only and
   controlled evaluation paths do not establish model certification.
2. Start from the public API and its checked example in
[Run and debug models](../../docs/model_testing.md).
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
