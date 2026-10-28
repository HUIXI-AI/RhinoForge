# Runtime configuration

[简体中文](runtime_config.zh.md) | English

RhinoForge has a small set of user-facing settings and a larger set of exact
model-profile and diagnostic controls. Set the complete environment before
importing `rpu_backend`; change environment settings only at the lifecycle listed in the table. Supported stage changes use the execution API described below.

This page lists controls from `python/rpu_backend`, `src` and Launch.

## How to read the tables

Parser abbreviations:

- **PB(default)** is the shared Python boolean parser. It strips whitespace and
  accepts case-insensitive `1`/`true`/`on` and `0`/`false`/`off`/empty; any other
  value raises `ValueError`.
- **B01(default)** is a boolean read by both Python and native code. Only literal
  `0` and `1` are portable and accepted by the Python mirror.
- **E1** means unset is off and only exact literal `1` enables the path.
- **N1** means unset is off and exact `1` or lowercase `true` enables the native
  path. Rows call out different native parsers explicitly.

Read/lifecycle abbreviations:

- **IMPORT**: package or native-extension initialization; changing it requires a
  fresh process.
- **NATIVE**: cached on first native use; changing it requires a fresh process.
- **MODEL**: bound while constructing a model, policy, or weights; create a new
  model. Because RhinoForge permits one live fused policy per process, a fresh
  process is the safe replacement procedure.
- **BUILD**: bound while a handle or Graph is built; clear and rebuild every
  affected Graph and its owning model, or use a fresh process.
- **CALL**: read for each applicable call. If it changes execution shape or Graph
  topology, the row still requires a rebuilt Graph.

Use `0`/`1` in deployment manifests even when a wider parser is documented.
Facade defaults below are applied only by that named public facade; the raw
source default applies to direct adapter use.

## Configuration and execution

Use `python -m examples.run --config PATH`. [Model examples](../examples/README.md) describes the model directories, formats and required inputs. General models use `[example]/[input]/[run]`; specialized policies use the model table shown in their templates. `example.opt_in` binds permitted controls before Torch import and rejects inherited conflicts. `[cpu].threads` applies before the first Torch import; thread aliases must agree.

Select main and expansion assets with `RPU_KERNEL_LIB_PATH` and `RPU_KERNEL_LIB_PATH_EXPANSION`. Keep asset paths and credentials in the deployment environment. `rpu_execution` configures model and component settings. Cold settings require a fresh model/process; supported stage updates use the [execution API](api_reference.md#execution-configuration-and-updates) between requests.

## Host execution settings

`run.warmup` defaults to two inference calls for every shared-runner target.
For a lazily captured fixed signature, these cover BUILD and one REPLAY.
Explicit counts, including zero and one for cold diagnostics, are honored
without hidden calls. `warmup_wall_ms_runs` records every warmup call separately
from `wall_ms_runs`. When `warmup >= 2`, `warmup_decode_steps` must be omitted or
equal `input.decode_steps`; shortened decode warmup is rejected before model
loading. Only explicit `warmup=0` or `1` permits a shortened cold diagnostic.

The shared examples runner prepares every model under `no_grad()` with
`inference_mode(False)`. Persistent weights and mutable Graph inputs therefore
retain version counters; `run.inference_mode` still controls warmup and timed
inference. Lazy component installation needs the same local preparation scope.
For caches keyed by tensor versions, an inference tensor has no usable counter:
refresh its derived data or compare owned content instead of assuming that an
unchanged tensor object has unchanged values.

These settings belong to the application process and its installed CPU runtime,
not to the device planner. The canonical `[pi05]` examples bind the allocator values
below before importing Torch, through `run.py`.
`run.torch_num_threads` defaults to eight. Library inference does not change
these process settings. Use a fresh process when changing them.

| Variable | Recommended Pi value | Read / change | Scope, effect, and risk |
|---|---:|---|---|
| `OMP_NUM_THREADS` | `8` | CPU runtime initialization / **IMPORT** | Bounds OpenMP host work; actual behavior depends on the installed Torch build. Explicit Torch thread setters take precedence. |
| `MKL_NUM_THREADS` | `8` | CPU runtime initialization / **IMPORT** | Bounds MKL host work when the installed CPU runtime uses MKL. |
| `MIMALLOC_PURGE_DELAY` | `1000` | mimalloc initialization / **IMPORT** | Upstream Pi host setting; delays purging freed pages in mimalloc-enabled builds, potentially retaining more host memory. |
| `MIMALLOC_ARENA_PURGE_MULT` | `0` | mimalloc initialization / **IMPORT** | Upstream Pi host setting; changes arena purge policy in mimalloc-enabled builds. Runtime versions determine its behavior. |

Qwen3, Qwen3-VL, Qwen3.5 and Pi0.5 accept cold `model.num_cores` and
component `linear_acc32` controls. A core budget is an integer `4`, `6` or `8`;
each adapter further narrows it by checkpoint, quantization and input profile.
Optimized Pi profiles use eight cores; reduced generic Pi profiles and quantized Qwen profiles follow their explicit model/configuration admission. `linear_acc32` is a boolean: `false` selects ACC16 and `true`
selects ACC32. Qwen3 and Qwen3.5 text expose it under `prefill`; image-text
entries also expose `vision`; Pi exposes `prefill`, `vision` and `action`.
Qwen3-VL additionally accepts the cold boolean `prefill.fast_replay`,
not `vision.fast_replay` or a separate decode table. Ordinary FP16 and quantized
API entries default to `false` when it is omitted; the applicable FP16 and
quantized example templates explicitly select `true`. Specialized profile
policies and native owner/Graph checks remain authoritative. These settings bind before
weight installation; close the model and load a new instance to change them.

Omitting `model.num_cores` selects eight. For admitted reduced-core dense Qwen3, Qwen3-VL and Qwen3.5 profiles, a budget of four uses four attention/MLP/head cores; a budget of six uses four attention/head cores and six MLP cores. Their image tower uses four. Construct caches through the model's cache factory so storage matches the selected topology. Supported sizes and quantized variants are listed in each [configuration guide](../examples/configs/CONFIGURATIONS.md).

Keep model switches scoped to that model's configuration or launch process. Explicit environment values override adapter `setdefault` values; the example runner applies its selected configuration before adapter import. Logging, allocator and Launch initialization settings must be present before importing the backend. Use the model template's opt-ins and defaults instead of exporting optimization flags globally.

Prepare Qwen3.5-35B-A3B mixed-E4M3 weights with the
[official-source download and CPU conversion commands](../examples/configs/qwen3_5/README.md#cmd-moe-mixed).
Text/Vision, core-count and ACC variants share the same converted checkpoint;
these runtime settings do not select a different conversion recipe.

## General and cache settings

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_BACKEND_SO` | Unset; explicit native-extension file path | Extension resolution / **IMPORT** | Takes precedence over packaged and source-build candidates. The file must exist and match this Python/PyTorch ABI. Set only to a trusted build. |
| `RPU_KERNEL_LIB_PATH` | Path to `rhinoOpLib_current.ref` and its adjacent `.kernels` | First asset use / **IMPORT** | Select the main asset matching this source. |
| `RPU_KERNEL_LIB_PATH_EXPANSION` | Expansion path; defaults to `rhinoExpansionOpLib_current.ref` beside main | First asset use / **IMPORT** | Select the matching expansion asset and adjacent `.kernels`. |
| `RPU_MODEL_CACHE` | `~/.cache/rhinoforge/models`; directory path | Package bootstrap for HF defaults / **IMPORT**, alias resolution / **CALL** | Root for RhinoForge model aliases. An explicit value also supplies HF cache defaults; a post-import change affects later aliases but does not reliably reconfigure already imported HF components. |
| `RPU_LOG_LEVEL` | `3`; decimal integer `0..5`; invalid values warn and fall back to `3` | Native-extension load / **IMPORT** | Process logging: `0` silent through `5` trace. High levels add output and may expose paths or request metadata. |
| `RPU_WARMUP` | `0`; integer, invalid becomes `0`, negative clamps to `0` | Adapter construction / **MODEL** | Warmup forward count for adapters that support it. Adds startup work and can consume diagnostic budgets. |
| `RPU_CAUSAL_PREFILL_PADDING_BUDGET` | `64`; non-negative decimal integer | Eligible causal prefill planning / **CALL**; rebuild Graph for a new plan | Maximum optional rows for the generic Qwen3/Llama planner. `0` disables optional padding; a changed plan changes Graph signatures. |
| `LKN_MAX_BATCH_ENTRIES` | `65536`; positive decimal, invalid/empty/zero uses default, cap `4194304` | Launch initialization and Graph planning / **IMPORT** | Process capacity for batched submissions. Too small rejects a Graph; excessive values reserve more host memory. |
| `LKN_KD_BUF_MB` | `8`; positive MiB, invalid/empty/zero uses default, cap `256` | Launch initialization and Graph planning / **IMPORT** | Process command-buffer capacity. Too small rejects a Graph; excessive values increase memory use. |
| `LKN_INSTR_BUF_MB` | `64`; positive MiB, invalid/empty/zero uses default, cap `1024` | Launch initialization and Graph planning / **IMPORT** | Process execution-buffer capacity. Too small rejects a Graph; excessive values increase memory use. |
| `HF_HOME` | `<model-cache>/.hf`; directory path | Package bootstrap and HF initialization / **IMPORT** | HF cache root; explicit values are preserved. |
| `HF_HUB_CACHE` | Hugging Face default; directory path | Package bootstrap and HF initialization / **IMPORT** | Standard Hub cache; defaults to `<HF_HOME>/hub` when unset. |
| `HUGGINGFACE_HUB_CACHE` | Hugging Face default; directory path | Package bootstrap and HF initialization / **IMPORT** | Compatibility Hub cache variable, handled like `HF_HUB_CACHE`; keep both consistent. |

`RPU_LOG_LEVEL` controls backend logging at import; `LKN_LOG_LEVEL` separately
controls Launch logging. Set both before import to control startup output.
Runtime `set_debug_level()` changes backend logging only. Forked children inherit
the current level; spawned children inherit `RPU_LOG_LEVEL` from the environment,
so update it before spawning when propagating a runtime change.

Set all three `LKN_*` capacity values before importing the backend. A facade may choose
larger defaults before its first adapter import, but it cannot resize a Launch
runtime that is already loaded. Larger capacities do not make an unsupported
model profile supported.

## Model-specific public settings

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_QWEN3_SPM_KV_BY_MHA` | `1`; exact `0` or `1` | Qwen3 fused-handle construction / **MODEL** | Allows the certified short-prefill profile to keep transient K/V in SPM when the exact plan fits. `0` pins the DDR-cache attention path; unsupported shapes fall back to DDR automatically. |
| `QWEN3_5_QUANT_ALLOW_UNCERTIFIED` | `0`; exact literal `1` enables | Qwen3.5 quantized preflight / **MODEL** | Opts in to uncertified W8/W4 evaluation; checkpoint format, projection scope and numerical requirements still apply. |
| `QWEN3_5_TEXT_CHUNK` | `0` auto; decimal `0` or integer at least `64` | Qwen3.5 text installation / **MODEL** | Cold per-handle prefill cap. Invalid or smaller positive values fail; use `rpu_execution` for new integrations. |
| `QWEN3_5_TEXT_PADDING_BUDGET` | `64`; decimal integer `0..64` | Qwen3.5 text installation / **MODEL** | Cold optional-padding budget. It changes prefill plans and Graph signatures. |
| `RPU_QWEN3_5_FREE_HF_WEIGHTS` | **PB(true)** | Qwen3.5 weight installation / **MODEL** | `1` releases original HF weights after transformed copies exist; `0` retains them and increases host memory. |
| `RPU_PI05_SIGLIP_W8A16` | **PB(true)** for an int8/uint8 bundle, otherwise **PB(false)** | Pi0.5 SigLIP installation / **MODEL** | Selects the existing quantized projection path. It must match checkpoint dtype and numerical validation. |
| `RPU_PI05_SIGLIP_W8A16_SCOPE` | `all`; `all`/`full`, `row`/`safe`, `out`/`out_proj`/`o`, `fc2`/`mlp.fc2`, or `none`/`0`/`off`/`false`/empty | Pi0.5 SigLIP installation / **MODEL** | Chooses quantized projection groups. Unknown text fails; every scope is a distinct numeric profile. |
| `RPU_PI05_ADARMS_DENSE_W8A16` | **PB(true)** for a quantized expert, otherwise **PB(false)** | Pi0.5 AdaRMS installation / **MODEL** | Selects quantized AdaRMS dense weights. Mismatch with the loaded profile fails or changes numerics. |

Prefer the public loader or policy `rpu_execution` argument for chunk and
padding choices. It is validated before weight loading and exposed as an immutable snapshot; supported changes replace that snapshot through the execution API.

## Fail-closed evaluation gates

These gates default off and accept only exact literal `1`. They are read during
preflight or construction (**MODEL**) and require a fresh model/process to
change. Enabling one admits a controlled evaluation path; it does not certify
the path, relax numerical gates, or expand the supported-model matrix.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED` | `0`; **E1** | Qwen3.5/G0.5 Vision preflight / **MODEL** | Enables controlled evaluation of numeric-blocked Qwen3.5 Vision 2B/4B and generic G0.5 Vision.  |
| `QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED` | `0`; **E1** | Preflight / **MODEL** | Legacy-named gate for the exact Source-only Qwen3-VL 32B W8A16 candidate. It requires a matching reserved-buffer runtime and unchanged exact LKN defaults. |
| `RPU_LINGBOT2_ALLOW_UNVALIDATED` | `0`; **E1** | Preflight / **MODEL** | LingBot-VLA-V2 controlled profiles. |
| `RPU_INTERNVLA_N1_ALLOW_NUMERIC_BLOCKED` | `0`; **E1** | Preflight / **MODEL** | InternVLA-N1 legacy controlled evaluation. |
| `RPU_S2_SDPA_BF16` | Unset/empty/`0` only; literal `1` and all other values are rejected | Preflight / **MODEL** | InternVLA policy tripwire for an unavailable asset; leave off. |

## Shared advanced execution selectors

These are profile-owned. A facade may set a validated value before model
construction; ambient overrides can break graph, memory, or numerical
contracts.

### Collective execution

Residual reduction has no public route, chunk, ring, or two-stage selector.
The ordinary FP16 ring uses the installed consumer core count, with eight as the
default. Admitted reduced profiles use their bound topology; local-SPM,
mixed-precision and fixed-layout routes retain eight-core bounds. Partial
producers clear every consumer shard before writing the active prefix. See
[Automatic residual reduction](../knowledge/concepts/allreduce-routing.md).

### Graph and replay

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_DEEP_FAST_REPLAY` | Off; any non-empty value whose first character is not `0` enables | Fused-handle construction / **MODEL** | Skips audited setup work on full-body replay. A profile must prove every skipped input/layout remains valid. |
| `RPU_FASTREPLAY_SKIP_SYNC` | On; empty, `0`/`false`/`off` disables | Graph BUILD / **BUILD** | Allows the common Graph executor to omit the mutable-kernel scan only when the replay did not touch Graph-owned kernel parameters and every Launch register-state token still matches the prepared batch. Mutable DMA endpoints are resolved and patched independently on every replay, and all command writes are published before submission. A dirty witness, token change, or token-query failure conservatively performs the full scan. This is a diagnostic opt-out, not a model route. |
| `RPU_GRAPH_DEFER_TO_COPY` | No independent default; legacy alias accepts `auto`/empty, `0`/`off`/`false`, or any other non-empty value for forced on | First host-op gate use / **NATIVE** | Consulted only when `RPU_GRAPH_HOST_OP_DEFER_GATE` is unset. Avoid setting both. |
| `RPU_GRAPH_HOST_OP_DEFER_GATE` | `auto`; `auto`/empty, `0`/`off`/`false`, or any other non-empty value for forced on | First host-op gate use / **NATIVE** | Controls deferral of stable host inputs during capture. Forced-on with unstable storage risks stale data. |
| `RPU_SKIP_IDLE_RECORD_FUNCTION` | On; `0`/`off`/`false` disable | First Python graph scope / **MODEL** | Omits idle profiler scopes only while the profiler is off. It preserves scopes when profiling and otherwise reduces host overhead. |

### KV, linear, normalization, and scheduling

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_ADARMS_FUSED_BCAST` | Off; non-empty except `0`/`false`/`False` | Handle/Graph build / **BUILD** | Fuses profile-owned AdaRMS broadcasts. The owning layout must be contiguous and stable. |
| `RPU_KVINSERT_HYBRID3_V16` | Off; non-empty except `0`/`false`/`False` | Each KV insertion / **CALL**; rebuild Graph | Selects a three-part aligned/tail schedule. Shape and cache signatures must be revalidated. |
| `RPU_KVINSERT_HYBRID_V16` | Off; non-empty except `0`/`false`/`False` | Each KV insertion / **CALL**; rebuild Graph | Selects an aligned bulk/tail schedule. Facades may own it; arbitrary enablement changes graph topology. |
| `RPU_KVINSERT_V16` | On; exact `0` or `false` disables, all other values enable | First native use / **NATIVE** | Selects the aligned KV-insert route when admitted. Disabling changes scheduling and performance, not logical cache length. |
| `RPU_KVINSERT_V16_ANY_TP` | Off; non-empty except `0`/`false`/`False` | First native use / **NATIVE** | Allows the aligned route for additional whole-head parallel factors. Unsupported geometry can fail admission. |
| `RPU_LINEAR_ACC32` | Retired | Not read by production launchers | Set component `linear_acc32` in `rpu_execution`; Wall-OSS rejects the old name in `runtime_env`. |
| `RPU_RMSNORM_NEWTON` | Retired | Not read by production launchers | Wall-OSS rejects this name in `runtime_env`; remove it from production configurations. |
| `RPU_RMSNORM_VWARP` | Compatibility input; Hy-VLA strictly accepts empty/`0`/`off`/`false`, `16`, `32`, or `auto`, while Wall-OSS rejects this variable | Owner cold-plan admission / **NATIVE** | The owning adapter translates it into a frozen per-handle capability. Invalid Hy-VLA text fails before process claim; native handles do not re-read it. |

Ordinary direct-SPM RMSNorm freezes available payloads per handle and uses a
common `(rows, cols)` resolver: V32 for divisible-by-32 rows, then V16 for
divisible-by-16 rows, otherwise BASE. Canonical V16/V32 routes use grid 8.
`RPU_RMSNORM_VWARP` is a Hy-VLA VLM cold compatibility translator only; an explicit
unavailable route fails at installation. Mixed RMSNorm reads BF16 and emits FP16.

<a id="qwen35-vision-selectors"></a>
### Qwen3.5 Vision selectors

These selectors apply only after the numeric-blocked controlled-evaluation gate
has admitted Qwen3.5 Vision. They do not expand its Source-only status.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `QWEN3_5_VISION_CHUNK` | Auto; decimal integer, empty/non-positive means auto | First native vision plan / **NATIVE** | Caps Qwen3.5 Vision chunking. Positive overrides must remain in the admitted shape and memory envelope. |
| `RPU_VISION_MERGER` | **PB(true)** | Qwen3.5 Vision setup / **MODEL** | Enables the device merger path. `0` selects a fallback and changes performance/numerics. |
| `RPU_VISION_STEP0` | **PB(true)** | Qwen3.5 Vision setup / **MODEL** | Enables the device first-stage path. `0` selects a fallback and changes performance/numerics. |

<a id="gr00t-profile"></a>
## GR00T profile

The builder treats base-zero-shot DROID and finetuned DROID as different exact
profiles. The former has two cameras at temporal offsets `[-15,0]` (four
images); the latter has two cameras at `[0]` (two images). Both require
embodiment 24. Profile, checkpoint revision, processor/statistics identity and
the CPU input envelope are checked before model construction, as is the exact
eight-file Cosmos manifest. A generic Qwen3-VL-2B alias is not a valid
substitute. Frozen processor input-oracles are S277/S145, their KVPAD16 floors
are 288/160, and the profile-local candidate cache capacities are 320/256.
The shared 512-row allocation and upstream 1024-row action-position table are
not input-support limits; single-chunk SPM and lifecycle validation still must
pass.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_GR00T_PARTIAL_MROPE` | Raw **PB(false)**; facade **PB(true)** | GR00T construction / **MODEL** | Uses precomputed partial multimodal position tables. |
| `RPU_GR00T_W8A16` | **PB(false)** | GR00T weight installation / **MODEL** | Quantizes the admitted backbone and action projections and changes numerics. |

## Hy-Embodied profile

Before either the direct builder or `HyEmbodiedPolicy` claims process ownership,
it combines builder defaults, the facade dtype, and `runtime_env` overrides into
one immutable cold plan. RMS, component, persistence, boolean, and quantization
selectors marked strict below reject malformed values before allocator policy
selection, terminal poisoning, or safetensors tensor reads. Weight layout,
native handles, cache geometry, and Graph signatures take their decisions only
from that plan and do not re-read those selectors from the environment. Strict
tokens are whitespace-trimmed and case-insensitive; commas delimit component
lists. An explicit empty component selector disables all components, while a
comma-only empty list fails. Use a fresh process to switch policy profiles. The
fixed three-camera facade accepts `prefix_len` only in `{192, 208, 224, 240}`.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_HY_VLA_ACTION_MLP_MC` | Unset means 8 cores; `0`/`off`/`false` means 1, all other values mean 8 | Cold-plan admission before process claim / **MODEL** | Selects multi-core action-MLP execution. It changes reduction order and must retain profile numerical gates. |
| `RPU_HY_VLA_ATTN_TP8` | Raw off; only `1`/`true`/`on` enables; builder default `1` | Cold-plan admission before process claim / **MODEL** | Replicates KV heads for eight-core attention. Cache geometry and weights use the frozen value. |
| `RPU_HY_VLA_CACHING_ALLOC` | On; strict `0`/`off`/`false` or `1`/`on`/`true`; an explicit empty value is on; other values fail | Cold-plan admission before process claim / **PROCESS** | Selects the common cold tensor-allocation policy before materialization. The first policy declaration is immutable for the process. It changes memory retention; call `empty_cache()` before measuring free memory. |
| `RPU_HY_VLA_DENOISE_UNROLL` | Builder default `1`; strict `0`/`off`/`false` or `1`/`on`/`true`; an explicit empty value is off; other values fail | Cold-plan admission before process claim / **MODEL** | Records the denoise loop as one profile-owned Graph. It changes Euler-state precision and numerical results. |
| `RPU_HY_VLA_FAST_REPLAY` | Builder default `1`; strict off/on aliases or an exact CSV subset of `vit`, `vlm`, `expert`; `all` means all; unknown components fail | Cold-plan admission before process claim / **NATIVE** | Skips audited layer-body emission on replay. Wrong use can leave dynamic inputs stale. |
| `RPU_HY_VLA_FAST_REPLAY_PRELOAD` | Builder default `1`; same strict component grammar as `RPU_HY_VLA_FAST_REPLAY` | Cold-plan admission before process claim / **NATIVE** | Also skips audited preload work on replay. It requires stable persistent state. |
| `RPU_HY_VLA_FUSED_MERGER` | Builder default `1`; strict `0`/`off`/`false` or `1`/`on`/`true`; an explicit empty value is off; other values fail | Cold-plan admission before process claim / **MODEL** | Folds the vision merger into the graph. The frozen Python decision is reused by the runner. |
| `RPU_HY_VLA_KVPAD16` | Builder default `1`; strict off/on aliases or an exact CSV subset of `vit`, `vlm`, `expert`; unknown components fail | Cold-plan admission before process claim / **NATIVE** | Pads/masks expert KV rows. It changes cache layout and signatures. |
| `RPU_HY_VLA_MASK_ONCE` | Builder default `1`; strict off/on aliases or an exact CSV subset of `vit`, `vlm`, `expert`; unknown components fail | Cold-plan admission before process claim / **NATIVE** | Reuses an invariant mask upload. It is unsafe when mask contents or storage can change. |
| `RPU_HY_VLA_MERGER_IN_GRAPH` | Builder default `1`; strict `0`/`off`/`false` or `1`/`on`/`true`; an explicit empty value is off; other values fail | Cold-plan admission before process claim / **MODEL** | Places the enabled fused merger inside capture. Requires `RPU_HY_VLA_FUSED_MERGER=1`. |
| `RPU_HY_VLA_MOT_NORM_NOMERGE` | Default `both`; strict off aliases, `1`/`on`/`true`/`both`, or an exact CSV subset of `qkv`, `mlp`; an explicit empty value means `both`; other text fails | Cold-plan admission before process claim / **NATIVE** | Selects profile-specific norm/merge scheduling. It changes temporary-memory and Graph structure. |
| `RPU_HY_VLA_PARTIAL_ROPE` | Builder default `expert,vlm`; strict off/on aliases or an exact CSV subset of `vit`, `vlm`, `expert`; unknown components fail | Cold-plan admission before process claim / **NATIVE** | Uses precomputed partial position tables for selected subsystems. Scope mismatch changes position semantics. |
| `RPU_HY_VLA_PATCH_EMBED_IN_GRAPH` | Builder default `1`; strict `0`/`off`/`false` or `1`/`on`/`true`; an explicit empty value is on; other values fail | Cold-plan admission before process claim / **MODEL** | Captures device patch embedding. Disabling selects a different host/device boundary. |
| `RPU_HY_VLA_PATCH_EMBED_MC` | Unset means 8 cores; `0`/`false` means 1, all other values mean 8; builder default `1` | Cold-plan admission before process claim / **MODEL** | Selects multi-core patch embedding. Weight layout and native execution use the frozen core count. |
| `RPU_HY_VLA_PERSIST_HANDLES` | Builder default `1`; strict off aliases or an explicit empty value disable all, on aliases/`all` enable all, and an exact CSV subset of `vit`, `vlm`, `expert` selects owners; unknown or empty-only CSV lists fail | Cold-plan admission before process claim / **MODEL** | Keeps selected handles alive across calls. It increases retained memory and enforces one-policy ownership. |
| `RPU_HY_VLA_PREFIX_TEMPLATE` | On; strict `0`/`off`/`false` or `1`/`on`/`true`; an explicit empty value is on; other values fail | Cold-plan admission before process claim / **MODEL** | Enables the profile prompt template. Changing it changes token input, not merely performance. |
| `RPU_HY_VLA_PROJ1_IN_MERGER` | On; strict `0`/`off`/`false` or `1`/`on`/`true`; an explicit empty value is on; other values fail | Cold-plan admission before process claim / **MODEL** | Includes the first projection in the merger path. The runner reuses the frozen decision for layout and ownership. |
| `RPU_HY_VLA_Q_INPLACE` | Builder default `1`; strict `0`/`off`/`false` or `1`/`on`/`true`; an explicit empty value is off; other values fail | Cold-plan admission before process claim / **NATIVE** | Reuses the query buffer in place. It is safe only when the profile proves the prior value dead. |
| `RPU_HY_VLA_RMSNORM_PAD16` | Raw off; strict off/on aliases or an exact CSV subset of `vit`, `vlm`, `expert`; unknown components fail | Cold-plan admission before process claim / **NATIVE** | Pads selected normalization rows. It changes layout and Graph census. |
| `RPU_HY_VLA_SILU_MUL` | Builder default `expert`; strict off/on aliases or an exact CSV subset of `vit`, `vlm`, `expert`; unknown components fail | Cold-plan admission before process claim / **NATIVE** | Fuses the selected activation/multiply schedule. It needs per-scope numerical validation. |
| `RPU_HY_VLA_VIT_PACKED` | Builder default `1`; strict `0`/`off`/`false` or `1`/`on`/`true`; an explicit empty value is off; other values fail | Cold-plan admission before process claim / **MODEL** | Packs the vision-tower execution profile. It changes shape/layout contracts and requires rebuilt weights/Graph. |
| `RPU_HY_VLA_W4A16` | Off; strict off/on aliases or an exact CSV subset of `vit`, `vlm`, `expert`, `vlm_text`, `vlm_vision`; unknown or empty-only CSV tokens fail | Cold-plan admission before process claim / **MODEL** | Selects W4A16 per scope; W4 takes precedence over W8 on overlap. Unsupported ViT W4 fails before ownership or tensor reads. Numeric profile changes. |
| `RPU_HY_VLA_W8A16` | Raw off; facade maps `fp16`→`0`, `w8a16`→`all`, `w8a16-expert`→`expert`, `w8a16-expert-vlmv`→`expert,vlm_vision`, `w8a16-vlm`→`vlm`, `w8a16-vit`→`vit`, `w8a16-no-vlm`→`vit,expert`; otherwise uses the W4 row's strict grammar | Cold-plan admission before process claim / **MODEL** | Selects W8A16 per scope. It must match the named precision profile; malformed tokens fail before ownership or tensor reads. |
| `RPU_RMSNORM_VWARP` | Hy-VLA builder default `auto`; strict empty/`0`/`off`/`false`, `16`, `32`, or `auto`; other text fails | Cold-plan admission before process claim / **NATIVE** | Freezes the admitted VLM RMSNorm capability; native handles do not re-read the environment. |

<a id="internvla-n1-and-navdp-profiles"></a>
## InternVLA-N1 and NavDP profiles

The controlled InternVLA path also requires a caller-supplied SHA256-pinned
asset manifest through its Python API; no environment path replaces it.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_INTERNVLA_DINO_FINAL_NORM_RPU` | **PB(true)** | DINO tower construction / **MODEL** | Runs the final normalization on device. `0` changes the host/device boundary and timing. |
| `RPU_INTERNVLA_DINO_NORM_FOLD` | **PB(false)** | DINO tower construction / **MODEL** | Folds input normalization into patch weights. It changes installed weights and must be rebuilt. |
| `RPU_INTERNVLA_HOST_CACHE` | **PB(true)** | Backbone construction / **MODEL** | Enables host memoization and stable owners. Disabling increases repeated host work. |
| `RPU_NAVDP_DENOISE_UNROLL` | Off; stripped `1`/`true`/`True` enables, other values disable | NavDP action call / **CALL**; rebuild Graph | Captures the fixed denoise loop. It changes Graph topology and step execution. |
| `RPU_NAVDP_BATCH` | On; stripped `1`/`true`/`True` enables, other values disable | First unrolled action dispatch / **CALL**; rebuild Graph/model | Batches trajectories. When on, sample count is fixed after first dispatch; later mismatches fail. |
| `RPU_NAVDP_SPM_KV_BY_MHA` | `1`; exact `0` or `1` | NavDP fused-handle construction / **MODEL** | Allows the certified initial self-attention profile to use transient SPM K/V when its exact joint layout fits. `0` pins that attention to DDR; ineligible calls remain on DDR. |

<a id="lingbot-vla-v2-profile"></a>
## LingBot-VLA-V2 profile

Prefer `Lingbot2Policy.from_checkpoint(..., runtime_env=...)`. The facade
validates an exact allowlist, applies a full snapshot before adapter import, and
restores the shell environment on close; native state is still process-cold, so
use a fresh process for another profile.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_LINGBOT2_RUNTIME_GRAPH` | `0`; current examples `1`; literal `0/1` | Construction / **MODEL** | Retains Vision/Prefill/Action Graphs and native denoising. Rejects exact-Z2, W4 and debug routing combinations. |
| `RPU_LINGBOT2_SPARSE_EXPERTS` | `0`; current examples `1`; literal `0/1` | Construction / **MODEL** | Uses compact Top4 FP16/W8 experts. Rejects dense grouped storage, W4, exact-Z2 and debug routing/chunk overrides. |
| `RPU_LINGBOT2_ROUTED_GEMM` | `0`; literal `0/1` | Construction / **MODEL** | Alternative dense grouped-expert route; requires grouped storage and rejects sparse storage. |
| `RPU_LINGBOT2_PREFILL_KV_PAIR` / `RPU_LINGBOT2_PREFILL_QK_NORM_ROPE` | **PB(false)** | Construction / **MODEL** | Experimental Prefill fusion; requires runtime Graph and matching native capability. Current examples leave both off. |
| `RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE` | Raw/facade `0`; `1`/`true`/`True`/`on` | Expert construction / **MODEL** | Uses a direct indexed AdaRMS schedule. It conflicts with denoise unroll and requires expert replay. |
| `RPU_LINGBOT2_BASE_W8A16` | Raw `0`; **E1**; W8/W4 facades `1` | Weight conversion / **MODEL** | Quantizes shared/base expert projections. It is part of the complete precision profile, not an independent toggle. |
| `RPU_LINGBOT2_DENOISE_UNROLL` | Raw `0`; facade `1`; facade normalizes only exact `1` to on | Policy construction / **MODEL** | Records the fixed ten-step denoise Graph and uses FP16 Euler state. `0` restores the host loop and changes numerics. |
| `RPU_LINGBOT2_ENCODER_1THREAD` | `1`; `1`/`true`/`True`/`on` enables | Policy construction / **MODEL** | Runs small host encoder work with one thread. `0` uses the caller's process-wide thread setting. |
| `RPU_LINGBOT2_EXPERT_REPLAY` | Facade `1`; `1`/`true`/`True`/`on` | Policy construction / **MODEL** | Enables stable expert Graph replay. It requires stable input owners; disabling increases build/dispatch work. |
| `RPU_LINGBOT2_EXPERT_W4A16` | Raw `0`; **E1**; W4 facade `1` | Weight conversion / **MODEL** | Selects W4A16 routed-expert weights. It wins over W8 if both are set; use only the complete W4 profile. |
| `RPU_LINGBOT2_EXPERT_W8A16` | Raw `0`; **E1**; W8 facade `1` | Weight conversion / **MODEL** | Selects W8A16 routed-expert weights unless W4 is on. Numeric profile changes. |
| `RPU_LINGBOT2_FP16_TOP4` | Native default on unless dense-router diagnostic is enabled; facade `1`; exact `0`/`1` | Expert weight installation / **MODEL** | Keeps strict FP16 top-4 routing. Disabling without the dense diagnostic fails; routing changes are accuracy-sensitive. |
| `RPU_LINGBOT2_GROUPED_EXPERTS` | Raw `0`; **E1**; W8/W4/FP16 facades `1` | Weight conversion and handle build / **MODEL** | Selects packed grouped experts. It must agree with weight format and row-chunk profile. |
| `RPU_LINGBOT2_HOST_PREFIX_OPT` | Facade `1`; `1`/`true`/`True`/`on` | Policy construction / **MODEL** | Keeps stable prompt-prefix owners and memoization. Required for VLM replay and direct-prefix output. |
| `RPU_LINGBOT2_LEGACY_PREPROC` | `0`; **E1** | Policy construction / **MODEL** | Compatibility override that forces legacy preprocessing. It changes model input and is not a performance-only switch. |
| `RPU_LINGBOT2_PREFILL_FAST_REPLAY` | Raw `0`; `1`/`true`/`True`/`on`; facade `1` | Prefill handle setup / **MODEL** | Enables audited prefill body replay skips. Requires VLM replay and stable prefix storage. |
| `RPU_LINGBOT2_PREFILL_W4A16` | Raw/facade `0`; `1`/`true`/`True`/`on` | Weight conversion / **MODEL** | Restricted option quantizing only admitted prefill projections to W4A16. Ordinary W4 uses prefill W8A16 instead. |
| `RPU_LINGBOT2_PREFILL_W8A16` | Raw `0`; `1`/`true`/`True`/`on`; W8/W4 facades `1` | Weight conversion / **MODEL** | Selects W8A16 prefill projections. It must be part of the complete precision profile. |
| `RPU_LINGBOT2_PREPROC` | `exact`; lower-case `exact`, `fast`, or `legacy`; other facade input normalizes to `exact` before build | Policy construction / **MODEL** | Selects preprocessing semantics and therefore changes model input. |
| `RPU_LINGBOT2_QWEN3VL_BASE` | Packaged/registry fallback; non-empty local directory containing a valid config | Policy construction / **MODEL** | Overrides the Qwen3-VL base assets. Wrong or untrusted contents fail validation; it is not a download credential. |
| `RPU_LINGBOT2_VISION_DIRECT_PREFIX` | Facade `0`; `1`/`true`/`True`/`on` | Policy construction / **MODEL** | Writes fused vision results into stable prefix owners. Requires host-prefix optimization, VLM replay, batching, and fused merger. |
| `RPU_LINGBOT2_VISION_DIRECT_PREFIX_NO_OUTPUT` | Facade `0`; `1`/`true`/`True`/`on` | Policy construction / **MODEL** | Suppresses the extra vision return only with direct-prefix enabled. Wrong pairing fails. |
| `RPU_LINGBOT2_VISION_FAST_REPLAY` | Raw `0`; `1`/`true`/`True`/`on`; facade `1` | Vision handle setup / **MODEL** | Enables audited vision replay skips. It requires the named vision topology and stable dynamic inputs. |
| `RPU_LINGBOT2_VISION_W8A16` | Raw `0`; `1`/`true`/`True`/`on`; W8/W4 facades `1` | Vision weight conversion / **MODEL** | Selects W8A16 vision projections. It changes numerics and must match the complete checkpoint profile. |
| `RPU_LINGBOT2_VLM_REPLAY` | Defaults `1` when host-prefix optimization is on, otherwise `0`; `1`/`true`/`True`/`on` | Policy construction / **MODEL** | Enables cross-call VLM prefill replay. `1` with unstable prefix storage is rejected to prevent stale reads. |

## Pi0.5 profile

The Pi0.5 loader owns these settings. Graph selectors must be fixed before
model construction and first capture.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_PI05_ADARMS_GRAPH` | **PB(true)** | AdaRMS construction / **MODEL** | Enables the AdaRMS GraphCache path. `0` uses the eager fallback and changes timing/lifecycle. |
| `RPU_PI05_ADARMS_W8A16_GRAPH` | **PB(true)** | Quantized AdaRMS construction / **MODEL** | Enables GraphCache for the W8A16 AdaRMS path. It must match quantized weights. |
| `RPU_PI05_DENOISE_GRAPH` | **PB(true)** | Action construction / **MODEL** | Captures denoise steps. In a v1.2.0 RhinoForge build without the current source patch, changing logical prefixes repeatedly BUILD Graphs until arenas are exhausted, while `0` loses the physical descriptor in the host loop and corrupts steady-state numerics. Neither is a production workaround before patching; see the [known issues and fixes](pitfalls.md#pi05-v120-denoise-known-issues-2026-10-22). |
| `RPU_PI05_DENOISE_UNROLL` | Raw **PB(false)**; Pi adapter supplies `1` | Action construction / **MODEL** | Records the fixed denoise loop in one Graph and uses FP16 Euler state. It changes numerics versus the host loop. |
| `RPU_PI05_EMBED_PREFIX_PATCH` | **PB(true)** | Adapter class-patch installation / **MODEL** | Enables the supported prefix-embedding patch. Change it only in a fresh process because class hooks are process state. |
| `RPU_PI05_EULER_FP16` | **PB(false)** | Each action call / **CALL**; rebuild prepared Graph profile | Uses FP16 rather than host-FP32 Euler state. It directly changes action numerics. |
| `RPU_PI05_FUSED_DENOISE` | **PB(true)** | Action construction / **MODEL** | Uses the fused denoise subsystem. `0` selects the legacy per-step path and changes topology/performance. |
| `RPU_PI05_GEMMA_GRAPH` | **PB(true)** | VLM construction / **MODEL** | Enables the Gemma GraphCache path. It requires stable prefix/cache owners. |
| `RPU_PI05_KEEP_CPU` | Unset/empty only; any non-empty value, including `0`, raises | Pi0.5 installation / **MODEL** | Retired-path tripwire. CPU weight retention through this variable is intentionally unsupported. |
| `RPU_PI05_KVINSERT_PAD16` | **B01(false)** | Expert construction / **MODEL** | Pads and masks KV rows. It changes cache layout and Graph signatures. |
| `RPU_PI05_PREFIX_MASK_CACHE` | **PB(true)** | Each action call / **CALL** | Reuses a one-entry host mask cache. `0` recomputes it; correctness still requires stable Graph inputs. |
| `RPU_PI05_PREFIX_PAD16` | **PB(true)** | Prefix planning / **CALL**; rebuild Graph for a new plan | Adds masked prefix padding for admitted shapes. It changes execution length/signature, not logical prefix length. |
| `RPU_PI05_SIGLIP_BATCH` | **PB(true)** | SigLIP construction / **MODEL** | Packs compatible camera inputs. Disabling changes Graph shapes and host/device traffic. |
| `RPU_PI05_SIGLIP_GRAPH` | **PB(true)** | SigLIP construction / **MODEL** | Enables the SigLIP GraphCache path. `0` is a diagnostic fallback with different timing. |

## Qwen3-VL shared Vision profile

These variables are consumed by Qwen3-VL Vision and by VLA facades that embed
that tower. Facades may set exact defaults before model construction.

The exact ordinary dense FP16 Qwen3-VL-2B, 4B, and 8B profiles use device patch
and merger GEMMs by default; merger LayerNorm stays on CPU in FP32.
Set `RPU_QWEN3VL_VISION_HOST_FP32_PATCH=1` before installation to retain
CPU FP32 patch projection and CPU FP32 mergers. This choice is snapshotted
at installation; later environment changes require a fresh model.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_QWEN3VL_VISION_BATCH` | **PB(false)** | Vision forward / **CALL**; rebuild Graph | Packs compatible images/views. It changes Graph signatures and memory; facade constraints still cap shapes. |
| `RPU_QWEN3VL_VISION_BATCH_CAP` | `3`; decimal integer clamped to at least `1` | Vision forward / **CALL**; rebuild Graph | Maximum compatible images per packed group. Larger values can exceed the profile memory envelope. |
| `RPU_QWEN3VL_VISION_FUSED_MERGER` | Raw off; **B01(false)** | Vision construction / **MODEL** | Folds the merger into the Vision Graph. Only literal `0`/`1` keeps Python and native readers consistent. |
| `RPU_QWEN3VL_VISION_HOST_FP32_PATCH` | **PB(false)** | Vision installation / **MODEL**; set before `to_rpu()`, fresh model to change | Snapshots the CPU FP32 patch profile. For exact ordinary dense FP16 VL2/VL4/VL8 it also retains CPU FP32 mergers. It changes numerics and transfer cost. |
| `RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE` | `0`; **E1** | Vision construction / **MODEL** | Moves patch embedding to device. It changes installed weights, numerics, and graph topology. |
| `RPU_QWEN3VL_VISION_ROPE_SPM` | Off; native value beginning with `1`, `t`, or `T` enables | First native vision use / **NATIVE** | Keeps vision position tables in SPM. It changes persistent memory use and requires a rebuilt vision handle. |

## RhinoVLA profile

These are source-level integration controls. Use the model repository's
runtime factory and an exact validated configuration; a RhinoForge switch by
itself is not an end-to-end contract.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_RHINOVLA_DENOISE_STATIC_CONTEXT_CACHE` | Off; non-empty except `0`/`false`/`False` | First native action use / **NATIVE** | Caches denoise context proven static by the factory. Wrong use reuses stale observations. |
| `RPU_RHINOVLA_FOLD_ACTION_TIME_IN` | Off; non-empty except `0`/`false`/`False` | First native action use / **NATIVE** | Uses the profile's folded action/time input projection. Requires matching installed weights. |
| `RPU_RHINOVLA_FUSED_ADARMS_GEMV` | Off; **B01(false)** | Action construction (**MODEL**) and first native use (**NATIVE**); fresh process | Enables fused AdaRMS projection. It must agree across Python/native state and changes the graph census. |
| `RPU_RHINOVLA_FUSED_SILU_MUL` | Off; **B01(false)** | Action construction (**MODEL**) and first native use (**NATIVE**); fresh process | Enables fused activation/multiply. It changes numerical ordering and needs profile validation. |
| `RPU_RHINOVLA_GATED_NO_SUB` | Off; **B01(false)** | Action construction (**MODEL**) and first native use (**NATIVE**); fresh process | Uses the profile's subtraction-free gated formulation. Enable only where equivalence was validated. |
| `RPU_RHINOVLA_KVINSERT_HYBRID_V16` | Off; non-empty except `0`/`false`/`False` | Each KV insertion / **CALL**; rebuild Graph | RhinoVLA-scoped alias that opts into the shared hybrid KV schedule. It changes cache Graph topology. |
| `RPU_RHINOVLA_PRECOMPUTE_ADARMS` | Off; non-empty except `0`/`false`/`False` | First native action use / **NATIVE** | Uses factory-bound precomputed AdaRMS tables. Missing/mismatched tables fail or produce stale conditioning. |
| `RPU_RHINOVLA_PRECOMPUTE_TIME_PROJ` | Off; non-empty except `0`/`false`/`False` | First native action use / **NATIVE** | Uses factory-bound precomputed time projections. It is valid only for the bound timestep schedule. |
| `RPU_RHINOVLA_SKIP_ADARMS_GEMV` | Off; **B01(false)** | Action construction (**MODEL**) and first native use (**NATIVE**); fresh process | Skips AdaRMS projection only when precomputed values are bound. Wrong combinations produce invalid conditioning. |
| `RPU_RHINOVLA_VISION_BATCH_MERGERS` | **PB(false)** | Vision installation / **MODEL** | Batches compatible merger calls. It changes Graph shape and memory use. |
| `RPU_RHINOVLA_VISION_BATCH_VIEWS` | **PB(false)** | Vision forward / **CALL**; rebuild Graph | Packs equal-size views and is required for some batched merger profiles. Mixed geometry falls back or fails. |
| `RPU_RHINOVLA_VISION_CPU_MERGER_NORM_FP16` | **PB(false)** | Vision installation / **MODEL** | Uses FP16 for the CPU merger-normalization fallback. It directly changes numerical results. |
| `RPU_RHINOVLA_VISION_PREP_CACHE` | **PB(false)** | Vision installation / **MODEL** | Memoizes stable preprocessing artifacts. It retains host memory and requires correct cache keys. |
| `RPU_RHINOVLA_VISION_RPU_MERGER_NORM` | **PB(false)** | Vision installation / **MODEL** | Runs merger normalization on device. It changes the boundary and numerical profile. |
| `RPU_RHINOVLA_VISION_RPU_MERGER_OUTPUT_RPU` | **PB(false)** | Vision installation / **MODEL** | Keeps merger output on device. Consumers must accept device-resident output and stable ownership. |
| `RPU_RHINOVLA_VISION_SKIP_RAW_SNAPSHOTS` | **PB(false)** | Vision installation / **MODEL** | `1` suppresses raw debug snapshots. Leaving it off adds memory/synchronization and may retain sensitive model inputs/intermediates. |

### Expert-only W8 construction

The trusted factory may pass `expert_w8a16` and `full_expert_w8a16` to
`RhinoVLAOnRPU` before weight installation. The first selects the seven expert
projections; the second additionally selects its AdaRMS condition projections
and requires the first. Both accept strict booleans or `None` (retain existing
environment selection). Public example `input.full_w8a16` maps to the constructor's
`full_expert_w8a16`, not to `RPU_RHINOVLA_FULL_W8A16`.

The exact full-expert mode binds 18 W8 condition owners and floating final-norm
precomputation with FP16 tables and action IO. The existing full-pipeline switch
still binds 19 W8 owners and six W8 IO scales, including its final norm. HIGH and
gate-TANH precomputation continue to require the full-pipeline mode. Precision,
IO owners and tables are cold state; changes require a new model. Neither scope
changes numerical acceptance criteria.

The separate full-pipeline examples explicitly enable `RPU_RHINOVLA_FULL_W8A16`
and check the installed text, vision, action IO and cold AdaRMS owners before
inference. The baseline template explicitly disables HIGH, gate-TANH
precomputation and vector Q/K norm; the HIGH template enables them. This scope
is distinct from expert-only W8; neither template is a numerical or task-quality
certification.

The pinned v3 `rtc=True` mode accepts the same FP16 and W8 precision scopes.
Use the corresponding `rtc_*.toml` factory template, or keep a controlled recipe
and select `run --rtc --delay 6`. Resident AdaRMS, expert fusions, HIGH, vector
normalization and partial RoPE retain their selected settings; RTC prepares
conditioning for each delay using that recipe. Precision and RTC mode are cold
state, while request delay may vary within `0..6`. See the
[RTC input contract](../examples/configs/rhinovla/v3/README.md#rtc).
With resident AdaRMS enabled, RTC loads the current denoising step's state,
committed-prefix and remaining-suffix condition tables into SPM and reuses them
across that step's 18 expert layers. This leaves room for retained Vision/Text
buffers; non-RTC retains its ten-step table residency.

RhinoVLA also accepts cold boolean `linear_acc32` in
`components.language_model.prefill` and `components.vision_encoder.vision`.
The default is ACC16 (`false`); `true` uses the existing text Linear or vision
block/fused-merger ACC32 path. Patch embedding, action and cold AdaRMS retain
their existing arithmetic. The fields cannot change after construction and are
not accepted for the action component. HIGH nonlinear/norm options are separate
from Linear accumulation. See the [v3 templates](../examples/configs/rhinovla/v3/README.md).

## Wall-OSS profile

`WallOssPolicy` supplies validated facade defaults before model construction.
Prompt selectors change the actual model input and must never be treated as
performance-only switches. With every selector unset, the pinned public
checkpoint follows `use_embodied_system_prompt_ratio=0.0` and uses the exact
plain `You are a helpful assistant.` system prompt from public `wall-x`.

Bind `camera_names` at construction and pass one image for each name on every
request. Vision chunking is selected through `rpu_execution.vision`; packed
Vision groups compatible image grids within `RPU_WALL_OSS_BATCH_MAX_SEQ`.
The RTC constructor binds its fixed camera/action profile and rejects component
`linear_acc32=true`; ordinary profiles use their explicit stage settings.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_WALL_OSS_ACTION_FP32_TAIL` | **PB(false)** | Each denoise call / **CALL**; rebuild prepared Graph profile | Uses host FP32 for the Euler tail. It changes action numerics and conflicts with fused/unrolled execution. |
| `RPU_WALL_OSS_ATTN_TP8` | Raw **PB(false)**; facade `1` | Weight installation / **MODEL** | Replicates KV heads for eight-core attention. Rebuild weights, caches, and Graphs. |
| `RPU_WALL_OSS_BATCH_MAX_SEQ` | `768`; decimal integer | Vision grouping / **CALL**; rebuild Graph | Caps packed vision sequence length. Too large can exceed memory; the selected public profile may further constrain it. |
| `RPU_WALL_OSS_BATCH_VISION` | **PB(true)** | Vision forward / **CALL**; rebuild Graph | Groups equal-grid images within the configured capacity; `0` uses per-image execution. |
| `RPU_WALL_OSS_DENOISE_UNROLL` | **PB(true)** | Action construction and calls / **MODEL** | Uses the fused denoise loop. `0` selects the host loop and changes topology and timing. |
| `RPU_WALL_OSS_DEVICE_PATCH_EMBED` | **PB(true)** | Vision construction / **MODEL** | Runs folded patch embedding on device. `0` uses CPU projection and changes boundary/performance. |
| `RPU_WALL_OSS_EXPERT0_DOWN_INT4` | **PB(true)** | Quantized LLM weight installation / **MODEL** | Uses the admitted compact expert-0 down projection. Some W4 profiles force it off; mismatches fail validation. |
| `RPU_WALL_OSS_FAST_IMGPROC` | **PB(true)** | Policy construction / **MODEL** | Uses the direct image-preprocessing path after a first-call parity check. `0` forces the HF processor. |
| `RPU_WALL_OSS_FAST_PROCESSOR` | **PB(true)** | Policy construction / **MODEL** | Bypasses the combined processor wrapper after a first-call parity check. `0` keeps the wrapper. |
| `RPU_WALL_OSS_FAST_REPLAY` | Raw off; non-empty not starting with `0` or `f`; facade `1` | Fused-handle construction / **MODEL** | Skips audited layer-body emission on replay. Wrong use can replay stale dynamic state. |
| `RPU_WALL_OSS_FUSED_ASSEMBLE` | Raw **PB(false)**; facade `1` | Vision construction / **MODEL** | Combines vision reorder with input-embedding construction. It requires the on-device embedding path. |
| `RPU_WALL_OSS_FUSED_DENOISE` | Facade **PB(true)** | Action construction / **MODEL** | Uses the fused action subsystem. `0` selects a legacy per-step path and changes topology/timing. |
| `RPU_WALL_OSS_GENERIC_PROLOGUE` | **PB(false)** | Prompt construction / **CALL** | Selects the generic prompt schema. It is mutually exclusive with other prologue selectors and changes tokens. |
| `RPU_WALL_OSS_HOST_CACHE` | **PB(true)** | Component construction / **MODEL** | Enables host layout/prompt memoization and stable owners. `0` increases repeated host work. |
| `RPU_WALL_OSS_KVINSERT_PAD16` | **PB(true)** | LLM construction / **MODEL** | Pads/masks prefill KV rows. It changes execution length, cache layout, and Graph signatures. |
| `RPU_WALL_OSS_MULTISUITE_PROLOGUE` | **PB(false)** | Prompt construction / **CALL** | Selects the multisuite prompt schema; mutually exclusive with generic/short modes. Changes tokens. |
| `RPU_WALL_OSS_ONDEVICE_EMBED` | **PB(true)** | Policy construction / **MODEL** | Builds token/vision embeddings on device. `0` uses the CPU fallback and disables the fused construction path. |
| `RPU_WALL_OSS_PARTIAL_MROPE` | **PB(true)**; facade sets `1` | LLM construction / **MODEL** | Uses the required Wall-OSS multimodal position-table path. Explicit `0` is rejected for admitted Wall-OSS input. |
| `RPU_WALL_OSS_PE_NORM_FOLD` | **PB(true)** | Policy construction / **MODEL** | Folds image normalization into patch weights when fast preprocessing is active. Rebuild installed weights to change it. |
| `RPU_WALL_OSS_SHORT_PROMPT` | **PB(false)** | Prompt construction / **CALL** | Selects the short prompt schema; mutually exclusive with other prologue modes. Changes tokens. |
| `RPU_WALL_OSS_VISION_FUSED_MERGER` | Raw **B01(false)**; facade `1` | Vision construction / **MODEL** | Folds the merger into the Vision Graph. Only literal `0`/`1` keeps Python/native state consistent. |
| `RPU_WALL_OSS_VISION_ROPE_SPM` | Raw off; **N1**; facade `1` | First native vision use / **NATIVE** | Keeps position tables in SPM. It changes persistent-memory use and requires a rebuilt vision handle. |

## Diagnostic-only inventory

The comprehensive public runner rejects every variable in this section when it
appears in TOML. A developer may still set one explicitly in the shell for a
controlled local investigation. These controls can change execution, dump model
data, expose tensor summaries or timing structure, or invalidate correctness/performance
claims. Leave them unset in production and review every generated artifact before
sharing it.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `QWEN3_5_VISION_GRAPH_DISABLE` | `0`; any value other than exact `0` disables | Vision forward / **CALL**; rebuild/clear Graph | Bypasses Qwen3.5 Vision GraphCache for controlled comparison. Replay and latency conclusions no longer apply. |
| `RPU_CHUNK_FORCE_UNSAFE` | Off; **N1** | First applicable native planner/launcher use / **NATIVE** | Bypasses a planner safety check. It can exceed execution constraints and must never produce deployable output. |
| `RPU_DYNAMO_MATERIALIZE_BREAKS` | **PB(false)** | Dynamo partitioning / **CALL**; recompile | Materializes partition breaks. It changes graph boundaries and adds transfers, so it is only a compiler diagnostic. |
| `RPU_GRAPH_DDR_SPM_LOG` | Off; non-empty whose first character is not `0` | First data-node execution / **NATIVE** | Logs node indexes, byte counts, and a bounded source checksum. Output is model-data-derived and logging changes timing. |
| `RPU_GRAPH_FORCE_ONESHOT_ON_REPLAY` | Off; non-empty whose first character is not `0` | First replay check / **NATIVE** | Executes a one-shot route instead of normal replay. It invalidates Graph lifecycle and performance conclusions. |
| `RPU_KVINSERT_V16_TRACE` | Off; any present value except exact `0` or `false`; an empty exported value enables | First native use / **NATIVE** | Logs KV route selection and shapes. It is diagnostic output, not proof of cache correctness. |
| `RPU_L2_BUFONLY` | Off by absence; any presence, including `0`, enables | Grouped-expert Graph emission / **BUILD** | Declares grouped buffers but runs the per-expert route for bisection. Changes graph and performance. |
| `RPU_L2_DOWN_ACC16` | Off; **E1** | Expert weight installation / **MODEL** | Forces the diagnostic ACC16 grouped-down route. It changes numerical results and is not a supported precision profile. |
| `RPU_L2_RCHUNK` | Runtime fallback `256`; positive decimal; supported grouped profiles require exact `1632` | Weight/profile validation (**MODEL**) and Graph emission (**BUILD**); fresh process | Overrides grouped scaling row chunks. Wrong values fail the sealed profile or change graph/timing. |
| `RPU_L2_ROUTED_ONLY` | Off; **E1** | Expert weight installation / **MODEL** | Isolates routed-expert contribution for bisection. Output is not full-model output. |
| `RPU_L2_SCHUNK` | `32512`; positive decimal, non-positive uses default | Graph emission / **BUILD** | Overrides grouped activation host chunking. It changes graph census and may reduce safety/performance. |
| `RPU_LINGBOT2_DEBUG_DENSE_SOFT_ROUTER` | Off; **E1** | Expert weight installation / **MODEL** | Replaces strict top-4 routing with dense soft routing. Accuracy is unvalidated and output is non-production. |
| `RPU_PI05_LOAD_NOISE` | Unset; path to an existing tensor `.pt`; missing path is ignored | Noise preparation / **CALL** | Replaces sampled noise after strict shape/finite checks using `weights_only=True`. Only trusted local files should be used. |
| `RPU_PI05_LOAD_PIXEL_VALUES` | Unset; path to an existing tensor `.pt`; missing path is ignored | Image feature call / **CALL** | Replaces processed pixel values after strict checks. It changes model input and may load sensitive test data. |
| `RPU_PI05_LOAD_PREFIX_EMBS` | Unset; path to an existing tensor `.pt`; missing path is ignored | Prefix preparation / **CALL** | Replaces prefix embeddings after strict checks. It bypasses normal upstream values and invalidates E2E claims. |
| `RPU_PI05_LOG_CONVERSION` | Off by absence; any non-empty value, including `0`, enables | Gemma prefill / **CALL** | Logs conversion/handle/chunk diagnostics. Adds output and may expose model shape/configuration. |
| `RPU_RHINOVLA_VISION_RPU_MERGERS_MEM_DEBUG` | **PB(false)** | Vision installation/materialization / **MODEL** | Prints allocator summaries around merger materialization. Adds synchronization and exposes memory structure. |
| `RPU_SIGLIP_ISOLATE_PATCH_EMBED` | Off; non-empty except `0`/`false`/`False` | First segment planning / **NATIVE** | Isolates Pi0.5 patch-embedding segments to diagnose graph-finalization failures. Changes graph segmentation. |
| `RPU_WALL_OSS_INSTRUMENT` | **PB(false)** | Wall-OSS forward stages / **CALL** | Prints host stage timings. Instrumentation overhead makes the same run unsuitable for clean latency reporting. |
| `RPU_WALL_OSS_PROLOGUE_FILLER` | Empty; arbitrary string | Prompt construction / **CALL** | Appends diagnostic content to the prompt. It changes tokens and makes output unsuitable for correctness claims. |
| `RPU_WALL_OSS_VISION_DEVICE_MERGED` | **PB(false)** | Vision construction / **MODEL** | Keeps the eager merged result on device as a diagnostic alternative to the normal fused route. Changes ownership/topology. |
| `WALL_OSS_PROFILE_PREFIX` | Unset; decimal integer target length | Each prediction / **CALL**; forbidden with prepared production Graphs | Truncates trailing text for a profiling shape after preserving image tokens. Resulting actions are meaningless. |

## Non-environment runtime controls

Prefer these public APIs over new environment switches.

### Per-handle `rpu_execution`

Public loaders and policies accept an immutable `rpu_execution` mapping. The
stage tables are `prefill`, `vision`, and `action`, with a separate `model`
control table for supported entries; each entry point
advertises the subset it supports and rejects every other stage or field before
weight loading.

| Field | Accepted value | Meaning and lifecycle |
|---|---|---|
| `model.num_cores` | Integer `4`, `6` or `8` | Cold compute budget, only for declaring entry points and admitted profiles. |
| `<stage>.linear_acc32` | Boolean | ACC16 (`false`, default) or ACC32 (`true`), for the components listed above. This does not change checkpoint quantization. |
| `prefill.fast_replay` | Boolean | Qwen3-VL only; ordinary API default `false`, explicitly enabled by applicable FP16/quantized examples. |
| `chunk_size` | `"auto"` or a positive integer multiple of 16 | Automatic selection or an exact chunk size for the named stage, within that model’s admitted range. Supported owners can update it through the execution API. |
| `padding_rows` | `"auto"` or a non-negative integer | Exact/automatic execution padding for the named stage. An exact integer is mutually exclusive with `padding_budget`. |
| `padding_budget` | Non-negative integer | Maximum optional padding considered by the stage planner. It cannot accompany an exact integer `padding_rows`. |

Pass the initial mapping to the loader/factory before weight installation and the first Graph BUILD. Owners with a live execution session may then update supported stage controls between requests using [`reconfigure_rpu_execution`](api_reference.md#execution-configuration-and-updates). Each supplied stage replaces its previous override and invalidates affected Graphs. Core count, accumulation precision, replay policy and other installation settings remain cold. Owners without a session require a new instance; every update must stay within the owner’s capability envelope.

### Graph snapshot policy

`GraphRuntimePolicy.batch_snapshot_mode` is a cold per-instance setting: `"auto"` uses snapshot replay when the SDK and operator assets support it; `"off"` keeps ordinary prepared Graph replay. Both retain capture, replay and the model's fast-replay setting.

```python
from dataclasses import replace
from rpu_backend.graph import GraphCache, GraphRuntimePolicy

policy = GraphRuntimePolicy.from_environment(batch_snapshot_mode="auto")
cache = GraphCache(runtime_policy=policy)
control_policy = replace(policy, batch_snapshot_mode="off")
```

Pass `graph_runtime_policy=policy` before weight installation to `Pi05Policy.from_pretrained`, `WallOssPolicy.from_pretrained` or `build_lingbot_vla_v2`. Wall component builders accept the same argument. Pi retains its execution core count and requires at least five Graph arenas; mirrored components reuse the first owner's cold policy. A supplied policy is complete and frozen; use `replace` when changing only snapshot mode.

AUTO checks Launch capabilities and the loaded asset's hash and program identity. Unknown assets, unsupported SDKs or rejected configuration fall back to ordinary replay. Model-specific environment flags cannot override these checks. Graph statistics expose `batch_snapshot_mode`, `batch_snapshot_available`, `batch_snapshot_effective_contracts` and `batch_snapshot_fallback_reason`; fallback reasons include `policy_off`, `sdk_symbols_unavailable`, `sdk_capability_unavailable`, `untrusted_program:<name>` and `sdk_configuration_rejected`.

### Memory and coherency controls

- `torch.rpu.set_caching_allocator(bool)` selects the process tensor allocator
  (default direct/off). The first explicit selection or non-empty device tensor
  allocation freezes the choice; repeating the same value is idempotent and a
  conflicting value raises. Use a fresh process to choose another mode.
  `torch.rpu.empty_cache()` releases cached, unused blocks; it cannot release
  live tensors or Graph-owned storage. Adapters that retain many tensors select
  the common caching policy before their first device allocation.
- `torch.rpu.memory_stats()` and `torch.rpu.get_memory_stats()` return
  address-free allocator counters. `caching_allocator_policy_frozen` reports
  whether selection is already immutable. When `caching_allocator_enabled` is true,
  `caching_allocator_mapping` counts live HostDDR segments owned by that
  allocator and `cached_idle_mapping` counts its fully idle cached segments.
  They exclude Launch, SPM, and direct allocations and are not a process-wide
  driver mapping count.
  `reset_peak_memory_stats()` resets peak counters, while
  `reset_accumulated_memory_stats()` resets accumulated allocation/free
  counters. Resetting counters does not free memory.
- `torch.rpu.set_ddr_flush(bool)` controls internal device-to-device flush points
  (default off). RhinoVLA preserves this process setting instead of enabling
  redundant flushes across its vision, text, and action components.
- `torch.rpu.set_ddr_flush_force(bool)` controls CPU/device boundary coherency
  (default on). Keep it enabled for normal inference; disabling it is unsafe
  outside an isolated microbenchmark. The matching `get_*` functions report
  current process state.

Cache maintenance belongs at actual CPU data accesses: publish a CPU write
once before device consumption, and synchronize device output before a CPU
read. device tensor views, static table bindings, and device-only handoffs do not
need another flush. Uploads and `.cpu()` copies perform the boundary operation.
An device-to-device `copy_` implemented with host `memcpy` still needs both read and
write boundary synchronization; the tensor device labels alone are insufficient.

Official Qwen3.5-27B W8/W4 selects shared caching allocation after acquiring model ownership and reserves four Graph arenas before weight upload. Original weights stay on CPU until each finalized layout is uploaded. Allocator conflicts release the owner before weight mutation. Text prefill uses a bounded one-shot Graph; decode retains its own Graph. W8 admits 4/6/8 cores and W4 group32 admits 8; see [27B configurations](../examples/configs/qwen3_5/README.md).

Synchronization covers the actual tensor byte range, including view offsets. Reused uploads still check current content and ownership; in-place input changes refresh the upload. A stable storage address does not imply unchanged contents.

### Profiling mechanisms

| Mechanism | What it measures | When to enable | Output/risk |
|---|---|---|---|
| `torch.rpu.set_profile(True)` | Backend wrapper and accumulated timing counters | After warmup if only steady-state counters are wanted; call `reset_profile_accumulators()` before the measured window | Lightweight console/counter diagnostics; it is not a PyTorch operation trace. |
| `torch.profiler.profile(...)` | CPU and `PrivateUse1` operation scopes, including Graph capture/replay ranges | Wrap only the calls to measure; include warmup separately according to the benchmark protocol | Produces an operation-level trace/table and adds profiler overhead. |
| `torch.rpu.hw_perf_trace(output_dir, max_dumps=32)` | Hardware trace from the matching sanitized Release Launch | Enable before model setup and Graph BUILD; the Qwen3 runner exposes `--hwperf` / `run.hwperf=true` | Writes Chrome hardware traces; bounds dump count and affects execution timing. Keep traces outside source/release artifacts. |

Runner support differs by entry; see [model profiling](model_testing.md#diagnose-a-call)
for the Qwen3 command and rejected latency-runner flags.

## Inventory maintenance

The tables contain one row for each current concrete environment/configuration
name. A new reader or facade preset must add a row in the correct tier; removing
the final reader/preset requires removing its row. Automated documentation
checks should compare the row set bidirectionally with source readers and keep
diagnostic variables below the stable `Diagnostic-only inventory` heading.

## Additional model and diagnostic settings

Set model switches before constructing the model; changing them requires a new
model instance. Defaults below apply when unset; templates can select different
values. Shape restrictions are enforced by the selected model and its
[configuration](../examples/configs/CONFIGURATIONS.md). Diagnostic outputs contain
values from the current invocation and stay in the caller's output directory.

| Variable | Unset/default | Accepted values | Effect |
|---|---|---|---|
| `HALO_CHUNK_DIAG` | unset | unset off; any set value enables | Log candidate SPM fit and the selected chunk in the shared planner; checked on each plan. |
| `HALO_NODE_HIST` | unset | unset off; any set value enables | Log captured node and kernel counts before enqueue; checked for each prepared batch. |
| `HALO_PROFILE` | 0 | only literal `1` enables | Profile HALO forwards with torch.profiler; read for each profiling context. |
| `HALO_PROFILE_DIR` | output/halo_profile | directory path | Write profiler tables and Chrome traces here when HALO_PROFILE is enabled. |
| `QWEN3_5_VISION_DBG_Q` | 0 | `1` or `true` enables | Capture each layer's rotated Q tensor when debug export is enabled; read on first use. |
| `QWEN3_VL_DELIVERY_ALLOW_NUMERIC_BLOCKED` | 0 | only literal `1` enables | Allow the fixed Qwen3-VL delivery profile through its admission gate; checkpoint and geometry checks still apply. |
| `RHINO_CHUNK` | 0 | integer | RhinoVLA prefill chunk fallback when rpu_execution is absent; positive values request an exact chunk, non-positive values leave automatic planning. |
| `RHINO_SMOKE_PREFIX_PADDING_ROWS` | 0 | integer | Diagnostic prefix-visibility truncation. The official descending-flow pipeline requires 0. |
| `RHINO_SPM` | unset | unset/empty off; any other value enables | Print SPM allocation summaries at RhinoVLA pipeline checkpoints; read per call. |
| `RHINO_TRACE` | unset | unset/empty off; any other value enables | Print RhinoVLA pipeline stage messages; read per call. |
| `RPU_GR00T_CROSS_KV_REUSE` | 0 | `0`/`1`; empty is off; other values rejected | Reuse cross-attention K/V across ordinary B1/Q41 four- or ten-step TP8 denoise bodies. Refresh the first body each inference; source geometry, cache capacity and full SPM admission apply. |
| `RPU_GR00T_CROSS_V_TRANSPOSE_REUSE` | 0 | `0`/`1`; empty is off; other values rejected | With cross-KV reuse and its raw-SPM route, transpose V once in the first body and reuse that bank in later denoise bodies. |
| `RPU_GR00T_FF1_RESIDENT_W8` | 0 | `0`/`1`; empty is off; other values rejected | Keep layer-0 FF1 W8 weights in SPM for the admitted ordinary B1/Q41 TP8 denoise loop. Requires its expansion kernel and full SPM admission; source-weight changes invalidate the owned copy. |
| `RPU_GR00T_KV_SPM_ONLY` | 0 | `0`/`1`; empty is off; other values rejected | Omit unused DiT DDR KV mirrors only where the ordinary four- or ten-step TP8 plan consumes raw SPM K/V; DDR, single-step and Z2 routes retain mirrors. |
| `RPU_GR00T_NATIVE_POSITION_CACHE` | 0 | `0`/`1`; empty is off; other values rejected | Cache versioned, content-keyed int32 position tables for ordinary unpadded VL calls. Padding and Z1 use their original position path. |
| `RPU_GR00T_PRECOMPUTE_ADALN` | 0 | `0`/`1`; empty is off; other values rejected | After weight installation, precompute the fixed four- or ten-step AdaLN tables for ordinary B1/Q41 TP8 unroll. Requires unchanged source owners and no prior forward or Z2 preparation. |
| `RPU_GR00T_PRECOMPUTE_TIME_GLUE` | 0 | `0`/`1`; empty is off; other values rejected | Precompute fixed timestep/output-modulation tables for the same ordinary TP8 denoise profile, independently of AdaLN precomputation. |
| `RPU_GR00T_QWEN3VL_SPM_Z1` | 0 | `1` or `true` enables | Select the exact FP16 one-image Vision→Text SPM path; rejects W8, partial rotary and host-FP32 patching. |
| `RPU_GR00T_RING_XOR3` | 0 | `0`/`1`; empty is off; other values rejected | Select the optional residual collective for admitted ordinary B1/Q41 TP8 FP16 or W8 O/FF2 producers; retains FP32 accumulation and final FP16 rounding. |
| `RPU_GR00T_SELF_QKV_MULTIBASE` | 0 | `0`/`1`; empty is off; other values rejected | Combine the ordinary W8 TP8 self-attention Q/K/V projections when the exact geometry and expansion kernel are available; cross-attention retains separate projections. |
| `RPU_GR00T_VL_KV_SPM_ONLY` | 0 | `0`/`1`; empty is off; other values rejected | Omit unused VL DDR KV mirrors for admitted ordinary B1 TP8 raw-SPM attention; DDR and prepared Z2 routes retain them. |
| `RPU_GRAPH_HCB_CHECKSUM` | 0 | unset/empty/leading `0` off; other values enable | Log shape, dtype, sums and small tensor values around aten::_to_copy callbacks; sampled on first callback execution, without addresses. |
| `RPU_GRAPH_TO_COPY_LOG` | 0 | unset/empty/leading `0` off; other values enable | Log stable-output registration and adopted-return byte counts; registration samples the flag on first use, capture reads it per call. |
| `RPU_INTERNVLA_NEXTDIT_FOLD_NORM_MOD` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Fold per-step AdaRMS modulation into fixed norm weights for fused unroll; per-step diagnostic execution keeps its original math. |
| `RPU_INTERNVLA_NEXTDIT_FUSED_QKQ` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Combine each NextDiT layer's self-Q, self-K and cross-Q projections; requires the exact admitted flow geometry. |
| `RPU_INTERNVLA_NEXTDIT_STEP_PREP_RPU` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | With folded norm modulation, move timestep projection, bias/tanh, folding and staging into the device denoise graph; SiLU remains on CPU. |
| `RPU_L2_CAPTURE_GATE` | 0 | only literal `1` enables | Capture LingBot2 gate and accumulator tensors; adds capture buffers at weight installation. |
| `RPU_L2_CAPTURE_INNORM` | 0 | only literal `1` enables | Capture LingBot2 layer-0 post-AdaRMS input; allocated at weight installation. |
| `RPU_L2_CAPTURE_L0` | 0 | only literal `1` enables | Capture LingBot2 layer-0 output; allocated at weight installation. |
| `RPU_L2_CAP_RESID` | 0 | only literal `1` enables | Capture LingBot2 per-layer inputs and attention residuals; allocated at weight installation. |
| `RPU_L2_DBG_PACKED` | 0 | only literal `1` enables | Enable the LingBot2 packed-weight diagnostic getter; checked on each getter call. |
| `RPU_L2_STAGES` | 0 | only literal `1` enables | Capture LingBot2 grouped intermediate stages; allocated at weight installation. |
| `RPU_LINGBOT2_DEBUG_DUMP_ROUTER_H` | 0 | only literal `1` enables | Capture the LingBot2 router input tensor; allocated at weight installation. |
| `RPU_LINGBOT2_MULTIVIEW_SPM_Z2` | 0 | only literal `1` enables | Select the exact three-camera 256×256 Z2 profile; requires RPU_LINGBOT2_ALLOW_UNVALIDATED=1 and the matching template selectors. |
| `RPU_LINGBOT_DENOISE_UNROLL` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Capture the ten-step LingBot Euler denoise loop as one graph. |
| `RPU_LINGBOT_ENCODER_1THREAD` | 1 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Use one CPU thread within the denoise loop and restore the previous count afterwards; vision keeps its existing thread setting. |
| `RPU_LINGBOT_EXPERT_REPLAY` | 1 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Reuse the action-expert graph across denoise steps with stable inputs and refreshed modulation; 0 rebuilds per step. |
| `RPU_LINGBOT_FUSED_ENCODER` | 1 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Precombine action/time encoder projections and per-step time bias; 0 retains the separate projection path. |
| `RPU_LINGBOT_ONDEVICE_ENCODER` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Run the fused suffix encoder, output projection and Euler update on the device. |
| `RPU_LINGBOT_VLM_REPLAY` | 1 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Reuse the VLM prefill graph across calls with the same prefix shape and a stable input buffer. |
| `RPU_P7_SPLIT_KERNELS` | empty | comma-separated names | Isolate matching kernel/operator names into separate Graph segments for diagnosis; read while preparing segments. |
| `RPU_PI05_DENOISE_GATEUP_RESIDENT_LAYERS` | 0 | literal `0`, `1` or `5` | Bundle the first one or five action layers' W8 Gate/Up weights for the exact P800/M50 TP8 ten-step loop. Requires prepared AdaRMS and body streaming; the separate GateUp-resident switch must be off. |
| `RPU_PI05_PREFILL_KV_ONLY` | 0 | `0`/`1`; empty is off; other values rejected | For the admitted paired TP8 Pi0.5 prefill profile, the final layer produces only K/V and omits unused hidden output. Requires the dedicated KV-only entry and ACC16 prefill; selected templates enable it. |
| `RPU_PI05_PREFILL_MASK_FP16` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Prepare and cache prefix additive masks in CPU FP16; dtype participates in the cache key. Selected paired TP8 templates enable it. |
| `RPU_PI05_PREFILL_ROPE_CACHE` | 0 | `0`/`1`; empty is off; other values rejected | Cache one position-indexed RoPE upload using source identity/version and positions. Native stable slots still refresh; paired Down/KV-direct/KV-only profiles require this off. |
| `RPU_PI05_PROBE_DIR` | unset | directory path | Write named Pi0.5 intermediate tensors and KV snapshots to the supplied directory; read per export, unset means no files. |
| `RPU_RHINOVLA_ADARMS_RESIDENT` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Preload precomputed AdaRMS tables into SPM for the exact v3 expert: ten-step residency for non-RTC; per-step state/prefix/suffix tables reused across 18 layers for RTC. Requires precomputed tables and rejects SKIP_ADARMS_GEMV. |
| `RPU_RHINOVLA_ALIGNED_KV` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Pad the exact v3 expert's physical prefix and KV writes while preserving logical masks and RoPE positions; checks capacity and clears padding. |
| `RPU_RHINOVLA_EXPERT_DEPTH_LIMIT` | unset | integer within checkpoint depth | Diagnostic expert depth in 1..checkpoint depth; adjusts prefix-layer selection. Official and exact W8 profiles reject reduced depth. |
| `RPU_RHINOVLA_EXPERT_FUSIONS` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Use norm/shift and gated-residual fusion in the exact v3 expert; requires precomputed AdaRMS, GATED_NO_SUB and both kernels. |
| `RPU_RHINOVLA_EXPERT_MLP_DIM_LIMIT` | unset | integer within checkpoint width | Diagnostic expert MLP width in 1..checkpoint width; official and exact W8 profiles reject a reduced width. |
| `RPU_RHINOVLA_EXPERT_W8A16` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Quantize the exact v3 expert Q/K/V/O and MLP weights with per-channel FP16 scales in non-RTC or RTC mode. Rejects reduced geometry and PACKED_QKV; full-pipeline W8 also enables expert W8. |
| `RPU_RHINOVLA_HIGH_PRECISION` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Select Newton RMSNorm and higher-precision activations for exact full-W8 v3 with resident precomputed AdaRMS. Requires matching optional kernels before capture. |
| `RPU_RHINOVLA_HIGH_PRECISION_FUSIONS` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | With HIGH_PRECISION, fuse action norm/shift and text/action SiLU×up using the corresponding higher-precision kernels. |
| `RPU_RHINOVLA_PACKED_QKV` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Combine immutable Q/K/V weights for the exact FP16 v3 expert, retaining original output planes. Requires new weight installation after changing this setting. |
| `RPU_RHINOVLA_PARTIAL_ROPE` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | With HIGH_PRECISION in exact full-W8 v3, select the existing partial-RoPE route over the full head while leaving the aligned padding row untouched. |
| `RPU_RHINOVLA_PRECOMPUTE_GATE_TANH` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | For full-W8 v3 with resident precomputed AdaRMS, evaluate fixed gate TANH while constructing timestep tables and reuse it in replay. |
| `RPU_RHINOVLA_VECTOR_K_NORM` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Use vector K normalization in exact full-W8 v3; requires HIGH_PRECISION and ALIGNED_KV, with tail clearing before normalization. |
| `RPU_RHINOVLA_VECTOR_Q_NORM` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | With HIGH_PRECISION in exact full-W8 v3, pad Q normalization to its vector shape and clear the added rows each layer/replay. |
| `RPU_RHINOVLA_VISION_INPUT_CACHE` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Reuse vision cast, patch-input and versioned position buffers; refresh pixels every call. Bound to the installed vision owner, with at most four entries. |
| `RPU_WALL_OSS_ACTION_KV_OWNER_FANOUT` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | For the exact non-resident action route, produce K/V once and distribute through SPM; M32 does not admit this route. |
| `RPU_WALL_OSS_ACTION_WEIGHT_RESIDENCY` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Preload eligible action weights once per graph: W8 layers1–4 or FP16 layers1–2. Base H32 requires sorting SiLU and FP16 transfer bit4 when using FP16. |
| `RPU_WALL_OSS_FP16_MLP_M224` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Split FP16 M640 Gate/Up into224+224+192 rows; M320 boundary calls retain their route. |
| `RPU_WALL_OSS_FP16_TRANSFER_MASK` | 0 | decimal `0..7` | Bits1/2/4 select exact FP16 Vision/Prefill/Action schedules; weights remain FP16. |
| `RPU_WALL_OSS_PREFILL_MIXED_PRECISION` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Select the RTC prefill FP16/BF16 boundary for the exact TP8 H2048/I11008/L36 profile; enable with Vision mixed precision. RTC templates set their explicit values. |
| `RPU_WALL_OSS_PREFILL_OWNER_DIRECT_KV` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Write K/V directly from the exact mixed, staged P640 owner-QKV route after partial RoPE. |
| `RPU_WALL_OSS_PREFILL_STAGED_MLP` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Select layer-major W8 P640/C320×2 MLP scheduling; requires mixed prefill, TP8, causal position0 and the exact RTC checkpoint. P672 retains its own schedule. |
| `RPU_WALL_OSS_RTC_ALLOW_NUMERIC_BLOCKED` | 0 | only literal `1` enables | Allow the named RTC FP16/W8A16 configurations through admission; W4 and unmatched profiles remain rejected. Does not set the other profile switches. |
| `RPU_WALL_OSS_RTC_EMBODIMENT` | X2 ex001_desktop | non-empty string | Robot name inserted into the RTC prompt; non-RTC calls use the dataset name. Supply the name matching the caller's robot mapping. |
| `RPU_WALL_OSS_SORTING_SILU_MUL` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Fuse SiLU×Mul for admitted sorting/base paths; exact FP16 routes also require their transfer bit. Paired small-image Vision can retain its dense mask across layers. |
| `RPU_WALL_OSS_VISION_LAYER_GROUP` | 0 | `0` or `32` | For exact three-image576-patch batched vision, keep each image through all vision layers. Mutually exclusive with weight-outer scheduling; process-pinned at construction. |
| `RPU_WALL_OSS_VISION_MIXED_PRECISION` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Select the RTC vision FP16/BF16 boundary with fused merger and576 patches per image; pair with prefill mixed precision. RTC templates set their explicit values. |
| `RPU_WALL_OSS_VISION_NATIVE_WINDOW_SDPA` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Use native window attention for the admitted W8 weight-outer profile or base two/three224 images with row reuse and SiLU; FP16 also requires transfer bit1. |
| `RPU_WALL_OSS_VISION_PAIR_252` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | For exactly three252 images, pair the first two vision inputs and retain the third separately. Requires generic window vision, fused merger, sorting SiLU and batch capacity>=672. |
| `RPU_WALL_OSS_VISION_ROW_WEIGHT_OUTER` | 0 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Reuse complete local weight slabs across vision row tiles for admitted compact/base shapes. FP16 requires transfer bit1; small-image mixed precision stays off. |
| `RPU_WALL_OSS_VISION_WEIGHT_OUTER` | 0; RTC W8 default1 | `0`/`1`, `false`/`true`, `off`/`on`; case-insensitive | Select the exact three-image576-patch schedule with batched vision, mixed precision and fused merger. FP16/W4 RTC force it off; incompatible with layer-group32 and pinned per process. |
| `RPU_WALL_OSS_W16_NEXT3_MASK` | 0 | `0` or `1` | Select the M160×4/N96 row schedule for prefill O projection. |
| `RPU_WALL_OSS_W16_NEXT4_MASK` | 0 | decimal `0..3` | Bit1 selects W8 M640 Gate/Up row scheduling; bit2 selects paired ring/postnorm. |
| `RPU_WALL_OSS_W16_REUSE_MASK` | 0 | `0` or `8` | Enable A16 reuse for resident action Gate/Up; other bit patterns are rejected. |
