# Runtime configuration

[简体中文](runtime_config.zh.md) | English

RhinoForge has a small set of user-facing settings and a larger set of exact
model-profile and diagnostic controls. Set the complete environment before
importing `rpu_backend`; changing a profile in a live process is unsupported
unless a row explicitly says that it is read per call.

This page documents every environment reader currently retained under
`python/rpu_backend` and `src`, plus the Launch capacity inputs. Listing a
variable does not make an experimental combination supported.

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

## Precedence and reproducible profiles

When using `examples/run_model.py`, each allowlisted value in `[runner.env]`
overwrites the inherited value of the same name before importing `torch` or
`rpu_backend`. Variables omitted from the table retain their shell value. A
model facade then uses `setdefault` for any profile-owned defaults, so the
effective process environment still wins; direct adapter use receives the raw
source defaults documented below.

Deployment-owned paths and credentials do not belong in TOML. In particular,
set `RPU_KERNEL_LIB_PATH` in the verified deployment environment, and pass
cold per-handle planning through the loader's explicit `rpu_execution`
argument. The model preflight remains authoritative and may reject an ambient
environment or execution setting outside the exact profile.

For a reproducible run, record the TOML hash and the effective values of every
explicitly set runtime variable. Use a fresh process when changing an IMPORT,
NATIVE, MODEL, or BUILD setting; do not infer that mutating `os.environ` after
construction reconfigured an existing handle or Graph.

## TOML parameter catalog

The [configuration index](../examples/configs/README.md) is the complete list of
current template paths. Each model uses one directory layout. Qwen3 is under
`qwen3/text`; Qwen3-VL separates `text`, `vision` and `vl`; Qwen3.5 separates
`text`, `vl` and its distinct legacy checkpoint format. LingBot exposes only
`v2`, RhinoVLA separates `v1`/`v3`, Wall-OSS uses `base`, and Pi0.5 uses
`2cam`/`3cam` with eight cores. Model names are not duplicated as loose TOML files.

### Model and policy configurations

| Table | Fields | Meaning |
|---|---|---|
| `[example]` | `profile_id`, `target`, `registry_alias`, `dtype`, optional `architecture` | Selects the exact catalog identity. The checkpoint loader checks actual metadata before inference. |
| `[example.opt_in]` | Exact profile-owned environment values | Applied before backend import; inherited conflicts fail. Values must match the selected catalog profile. |
| `[input]` | `checkpoint`, plus model-specific prompt/images/tensors and generation or policy options | An explicit checkpoint overrides the alias. Requests and files remain caller-owned; consult the model README. |
| `[run]` | `warmup`, `runs`, `seed`, `torch_num_threads`, `inference_mode`, `output_dir` | Controls repeated inference and host settings; loading and graph preparation occur before measured samples. |
| `[run]` | `profile`, `timing_mode`, `summary_statistic`, `warmup_decode_steps` where supported | Torch tracing is a separate diagnostic invocation. Generation timing fields have model-specific restrictions. |
| `[rpu_execution.model]` | `num_cores` | Cold compute budget, admitted by the selected model/profile. Pi example templates use eight cores. |
| `[rpu_execution.<stage>]` | `chunk_size`, supported booleans; prefill padding fields where admitted | Cold per-handle planning. Actual constructors validate each stage, field and shape. |
| `[rpu_execution.components.<component>.<stage>]` | Supported per-component controls | Composite model overrides; accepted component IDs and fields are checked before installation. |

`example.profile_id` refers to [profiles.json](../examples/configs/profiles.json),
which contains only identities used by current templates. It binds precision,
component and opt-in constraints, not model quality or performance certification.
Qwen generation uses `input.decode_steps` for calls after the first prefill
output token, and `input.prefill_tokens` for an explicit constructed input length.
These are not interchangeable with `max_new_tokens` in a caller-owned older file.

Pi0.5 uses `[pi05].precision` instead of `[example]`. Its `[input]` contains
`checkpoint`, `batch`, `cameras`, `text_tokens`, and `num_steps`; the runner binds
the corresponding public optimized policy profile and validates its paired
prefill geometry. See [Pi configurations](../examples/configs/pi05/README.md).

RhinoVLA's `input.runtime_factory`, `factory_config`, and `request` name a trusted
caller implementation, a JSON factory configuration, and a tensor-only request.
The selected version's factory defines its model-specific fields; RhinoForge
binds the checkpoint, step count and cold execution configuration. See
[RhinoVLA configurations](../examples/configs/rhinovla/README.md).
The v3 W8 examples additionally bind the boolean `input.expert_w8a16` and
`input.full_w8a16` fields to the expert constructor. Factory JSON must agree
exactly; the runner checks the bound expert's installed integer weights and
scales before inference. These fields describe the action expert's scope,
not the precision of the entire policy. The separate experimental full-pipeline
templates add an explicit environment selection and full installed-owner check;
they are not default recommendations or numerical certifications.

### Independent integration templates and caller-owned files

G0.5, GR00T, NavDP and SigLIP retain their own `[runner]/[model]/[request]`
integration templates. Existing caller-owned files in this format remain
accepted, without shipping a second copy of the same model configuration.

| Table | Fields | Meaning |
|---|---|---|
| `[runner]` | `target` | Selects the direct integration entry. |
| `[runner.env]` | Non-diagnostic variables documented on this page, excluding runtime asset paths and credentials | Applied before importing PyTorch or RhinoForge. Omitted names remain inherited. |
| `[runner.torch_profile]` | `enabled`, `output`, `record_shapes`, `profile_memory`, `with_stack` | Command-wide Torch trace settings. |
| `[model]`, `[request]`, `[generation]` | Fields accepted by the selected direct example | Model locations and caller-owned inputs. |
| `[runtime]` | Trusted RhinoVLA factory-defined fields in caller-owned older files | Passed through the public factory contract. |

Do not mix `[input]/[run]` into a `[runner/model/request]` file; it is rejected.
`--check-config` checks structure and cold constraints without loading Torch or
accessing a board. It does not inspect local weights or certify an input request.

`padding_rows` is `"auto"` or a non-negative integer. An exact value and
`padding_budget` cannot both be active. Chunk sizes are `"auto"` or positive
multiples of 16, further narrowed by each loader's actual profile. Full cold
planning semantics are documented under [non-environment runtime controls](#non-environment-runtime-controls).

## Host execution settings

The shared examples runner prepares every model under `no_grad()` with
`inference_mode(False)`. Persistent weights and mutable Graph inputs therefore
retain version counters; `run.inference_mode` still controls warmup and timed
inference. Lazy component installation needs the same local preparation scope.
For caches keyed by tensor versions, an inference tensor has no usable counter:
refresh its derived data or compare owned content instead of assuming that an
unchanged tensor object has unchanged values.

These settings belong to the application process and its installed CPU runtime,
not to the RPU planner. The canonical `[pi05]` examples bind the allocator values
below before importing Torch, through either `run_model.py` or `pi05.py`.
`run.torch_num_threads` defaults to eight. Library inference does not change
these process settings. Use a fresh process when changing them; older direct
example formats do not apply `[runner.env]` on behalf of the common runner.

| Variable | Recommended Pi value | Read / change | Scope, effect, and risk |
|---|---:|---|---|
| `OMP_NUM_THREADS` | `8` | CPU runtime initialization / **IMPORT** | Bounds OpenMP host work; actual behavior depends on the installed Torch build. Explicit Torch thread setters take precedence. |
| `MKL_NUM_THREADS` | `8` | CPU runtime initialization / **IMPORT** | Bounds MKL host work when the installed CPU runtime uses MKL. |
| `MIMALLOC_PURGE_DELAY` | `1000` | mimalloc initialization / **IMPORT** | Upstream Pi host setting; delays purging freed pages in mimalloc-enabled builds, potentially retaining more host memory. |
| `MIMALLOC_ARENA_PURGE_MULT` | `0` | mimalloc initialization / **IMPORT** | Upstream Pi host setting; changes arena purge policy in mimalloc-enabled builds. Runtime versions determine its behavior. |

Qwen3, Qwen3-VL, Qwen3.5 and Pi0.5 accept cold `model.num_cores` and
component `linear_acc32` controls. A core budget is an integer `4`, `6` or `8`;
each adapter further narrows it by checkpoint, quantization and input profile.
The optimized Pi examples and quantized Qwen examples retain eight-core
admission. `linear_acc32` is a boolean: `false` selects ACC16 and `true`
selects ACC32. Qwen3 and Qwen3.5 text expose it under `prefill`; image-text
entries also expose `vision`; Pi exposes `prefill`, `vision` and `action`.
Qwen3-VL additionally accepts the cold boolean `prefill.fast_replay`,
not `vision.fast_replay` or a separate decode table. Ordinary FP16 and quantized
API entries default to `false` when it is omitted; the applicable FP16 and
quantized example templates explicitly select `true`. Specialized profile
policies and native owner/Graph checks remain authoritative. These settings bind before
weight installation; close the model and load a new instance to change them.

## General and cache settings

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_KERNEL_LIB_PATH` | Combined operator asset beside the extension; existing file path with an adjacent `.kernels` manifest | First asset access / **IMPORT** | Process-wide asset selection. A missing, incompatible, or untrusted asset or manifest prevents safe execution. |
| `RPU_MODEL_CACHE` | `~/.cache/rhinoforge/models`; directory path | Package bootstrap for HF defaults / **IMPORT**, alias resolution / **CALL** | Root for RhinoForge model aliases. An explicit value also supplies HF cache defaults; a post-import change affects later aliases but does not reliably reconfigure already imported HF components. |
| `RPU_LOG_LEVEL` | `3`; decimal integer `0..5`; invalid values warn and fall back to `3` | Native-extension load / **IMPORT** | Process logging: `0` silent through `5` trace. High levels add output and may expose paths or request metadata. |
| `RPU_WARMUP` | `0`; integer, invalid becomes `0`, negative clamps to `0` | Adapter construction / **MODEL** | Warmup forward count for adapters that support it. Adds startup work and can consume diagnostic budgets. |
| `RPU_CAUSAL_PREFILL_PADDING_BUDGET` | `64`; non-negative decimal integer | Eligible causal prefill planning / **CALL**; rebuild Graph for a new plan | Maximum optional rows for the generic Qwen3/Llama planner. `0` disables optional padding; a changed plan changes Graph signatures. |
| `LKN_MAX_BATCH_ENTRIES` | `65536`; positive decimal, invalid/empty/zero uses default, cap `4194304` | Launch initialization and Graph planning / **IMPORT** | Process capacity for batched submissions. Too small rejects a Graph; excessive values reserve more host memory. |
| `LKN_KD_BUF_MB` | `8`; positive MiB, invalid/empty/zero uses default, cap `256` | Launch initialization and Graph planning / **IMPORT** | Process command-buffer capacity. Too small rejects a Graph; excessive values increase memory use. |
| `LKN_INSTR_BUF_MB` | `64`; positive MiB, invalid/empty/zero uses default, cap `1024` | Launch initialization and Graph planning / **IMPORT** | Process execution-buffer capacity. Too small rejects a Graph; excessive values increase memory use. |
| `HF_HOME` | Hugging Face default; directory path | Package bootstrap and HF initialization / **IMPORT** | Standard HF cache root. `RPU_MODEL_CACHE` supplies it only with `setdefault`; caller values win. |
| `HF_HUB_CACHE` | Hugging Face default; directory path | Package bootstrap and HF initialization / **IMPORT** | Standard Hub cache. An explicit `RPU_MODEL_CACHE` supplies `<HF_HOME>/hub` only when unset. |
| `HUGGINGFACE_HUB_CACHE` | Hugging Face default; directory path | Package bootstrap and HF initialization / **IMPORT** | Compatibility Hub cache variable, handled like `HF_HUB_CACHE`; keep both consistent. |

Set all three `LKN_*` values before importing the backend. A facade may choose
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
padding choices. It is validated before weight loading and is immutable after
binding.

## Fail-closed evaluation gates

These gates default off and accept only exact literal `1`. They are read during
preflight or construction (**MODEL**) and require a fresh model/process to
change. Enabling one admits a controlled evaluation path; it does not certify
the path, relax numerical gates, or expand the supported-model matrix.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED` | `0`; **E1** | Qwen3.5/G0.5 Vision preflight / **MODEL** | Enables controlled evaluation of numeric-blocked Qwen3.5 Vision 2B/4B and generic G0.5 Vision. It does not certify image output. |
| `QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED` | `0`; **E1** | Preflight / **MODEL** | Legacy-named gate for the exact Source-only Qwen3-VL 32B W8A16 candidate. It requires a matching reserved-buffer runtime and unchanged exact LKN defaults; it does not certify numeric, task, performance, or release-runtime gates. |
| `RPU_LINGBOT2_ALLOW_UNVALIDATED` | `0`; **E1** | Preflight / **MODEL** | LingBot-VLA-V2 controlled profiles; output is not robot certified. |
| `RPU_INTERNVLA_N1_ALLOW_NUMERIC_BLOCKED` | `0`; **E1** | Preflight / **MODEL** | InternVLA-N1 legacy controlled evaluation. |
| `RPU_S2_SDPA_BF16` | Unset/empty/`0` only; literal `1` and all other values are rejected | Preflight / **MODEL** | InternVLA policy tripwire for an unavailable asset; leave off. |

## Shared advanced execution selectors

These are profile-owned. A facade may set a validated value before model
construction; ambient overrides can break graph, memory, or numerical
contracts.

### Collective execution

Residual reduction has no public route, chunk, ring, or two-stage selector.
The shared wrapper always uses the generated eight-core ring and chooses its
schedule from validated geometry. Partial producers clear inactive shards
before reduction. See
[Automatic residual reduction](../knowledge/concepts/allreduce-routing.md).

### Graph and replay

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_DEEP_FAST_REPLAY` | Off; any non-empty value whose first character is not `0` enables | Fused-handle construction / **MODEL** | Skips audited setup work on full-body replay. A profile must prove every skipped input/layout remains valid. |
| `RPU_FASTREPLAY_SKIP_SYNC` | On; empty, `0`/`false`/`off` disables | Graph BUILD / **BUILD** | Allows the common Graph executor to omit the mutable-kernel scan only when the replay did not touch Graph-owned kernel parameters and every Launch register-state token still matches the prepared batch. Mutable DMA endpoints are resolved and patched independently on every replay, and all command writes are published before submission. A dirty witness, token change, or token-query failure conservatively performs the full scan. This is a diagnostic opt-out, not a model route. |
| `RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN` | Off; exact `1`, `true`, `True`, or `on` enables | First native coexistence use / **NATIVE** | Retains a persistent SPM generation across profile-owned subsystem handoff. Wrong ownership can corrupt later execution. |
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
| `RPU_LINEAR_ACC32` | Off; strict `0`/`1`, `false`/`true`, or `off`/`on` with listed case variants; invalid fails | First native use / **NATIVE** | Chooses FP32 accumulation for tiled linear families. It changes tiles, memory use, performance, and numerics. |
| `RPU_RMSNORM_NEWTON` | Off; strict `0`/`1`, `false`/`true`, or `off`/`on` with listed case variants; invalid fails | First native use / **NATIVE** | Selects refined normalization. It changes numerics and can change available schedules. |
| `RPU_RMSNORM_VWARP` | Compatibility input; Hy-VLA strictly accepts empty/`0`/`off`/`false`, `16`, `32`, or `auto`, while Wall-OSS rejects this variable | Owner cold-plan admission / **NATIVE** | The owning adapter translates it into a frozen per-handle capability. Invalid Hy-VLA text fails before process claim; native handles do not re-read it. |

<a id="qwen35-vision-selectors"></a>
### Qwen3.5 Vision selectors

These selectors apply only after the numeric-blocked controlled-evaluation gate
has admitted Qwen3.5 Vision. They do not expand its Source-only status.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `QWEN3_5_VISION_CHUNK` | Auto; decimal integer, empty/non-positive means auto | First native vision plan / **NATIVE** | Caps Qwen3.5 Vision chunking. Positive overrides must remain in the admitted shape and memory envelope. |
| `RPU_VISION_MERGER` | **PB(true)** | Qwen3.5 Vision setup / **MODEL** | Enables the RPU merger path. `0` selects a fallback and changes performance/numerics. |
| `RPU_VISION_STEP0` | **PB(true)** | Qwen3.5 Vision setup / **MODEL** | Enables the RPU first-stage path. `0` selects a fallback and changes performance/numerics. |

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
| `RPU_GR00T_KVPAD16` | Raw off; facade `1`; native `1` or a value beginning with `t`/`T` enables | First native KV setup / **NATIVE** | Pads and masks KV rows for the GR00T expert. It changes cache layout and Graph signatures. |
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
| `RPU_INTERNVLA_DINO_FINAL_NORM_RPU` | **PB(true)** | DINO tower construction / **MODEL** | Runs the final normalization on RPU. `0` changes the host/device boundary and timing. |
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
| `RPU_PI05_DENOISE_GRAPH` | **PB(true)** | Action construction / **MODEL** | Captures denoise steps. Disabling changes dispatch behavior and invalidates replay measurements. |
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

The exact ordinary dense FP16 Qwen3-VL-2B, 4B, and 8B profiles use RPU patch
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
| `RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE` | `0`; **E1** | Vision construction / **MODEL** | Moves patch embedding to RPU. It changes installed weights, numerics, and graph topology. |
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
| `RPU_RHINOVLA_VISION_RPU_MERGER_NORM` | **PB(false)** | Vision installation / **MODEL** | Runs merger normalization on RPU. It changes the boundary and numerical profile. |
| `RPU_RHINOVLA_VISION_RPU_MERGER_OUTPUT_RPU` | **PB(false)** | Vision installation / **MODEL** | Keeps merger output on RPU. Consumers must accept device-resident output and stable ownership. |
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

The Source-only facade pins Vision `chunk_size=768`, disables packed Vision,
and requires `preflight_images(images)` before checkpoint weight loading. The
CPU processor's frozen grid must be exactly two `[1,32,32]` images. Each image
is executed independently with the shared `768+256` KV-first plan; `auto`, any
other grid, and later grid changes fail before the process-wide claim or before
any Vision work, respectively. This is an admission contract, not board
validation evidence.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_WALL_OSS_ACTION_FP32_TAIL` | **PB(false)** | Each denoise call / **CALL**; rebuild prepared Graph profile | Uses host FP32 for the Euler tail. It changes action numerics and conflicts with fused/unrolled execution. |
| `RPU_WALL_OSS_ATTN_TP8` | Raw **PB(false)**; facade `1` | Weight installation / **MODEL** | Replicates KV heads for eight-core attention. Rebuild weights, caches, and Graphs. |
| `RPU_WALL_OSS_BATCH_MAX_SEQ` | `768`; decimal integer | Vision grouping / **CALL**; rebuild Graph | Caps packed vision sequence length. Too large can exceed memory; the selected public profile may further constrain it. |
| `RPU_WALL_OSS_BATCH_VISION` | Raw **PB(true)**; Source-only facade `0` | Vision forward / **CALL**; rebuild Graph | Direct builders may pack equal-grid images up to the cap. The public facade fixes per-image preflight and rejects enabling this selector. |
| `RPU_WALL_OSS_DENOISE_UNROLL` | **PB(true)** | Action construction and calls / **MODEL** | Uses the fused denoise loop. `0` selects the host loop and changes topology and timing. |
| `RPU_WALL_OSS_DEVICE_PATCH_EMBED` | **PB(true)** | Vision construction / **MODEL** | Runs folded patch embedding on RPU. `0` uses CPU projection and changes boundary/performance. |
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
| `RPU_WALL_OSS_ONDEVICE_EMBED` | **PB(true)** | Policy construction / **MODEL** | Builds token/vision embeddings on RPU. `0` uses the CPU fallback and disables the fused construction path. |
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
| `RPU_ALL_GATHER_FORCE_MULTI_CORE` | Off; **N1** | First native use / **NATIVE** | Forces the multi-core schedule for A/B comparison. It is not a generally supported performance selector. |
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
| `RPU_WALL_OSS_VISION_DEVICE_MERGED` | **PB(false)** | Vision construction / **MODEL** | Keeps the eager merged result on RPU as a diagnostic alternative to the normal fused route. Changes ownership/topology. |
| `WALL_OSS_PROFILE_PREFIX` | Unset; decimal integer target length | Each prediction / **CALL**; forbidden with prepared production Graphs | Truncates trailing text for a profiling shape after preserving image tokens. Resulting actions are meaningless. |

## Non-environment runtime controls

Prefer these public APIs over new environment switches.

### Cold per-handle `rpu_execution`

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
| `chunk_size` | `"auto"` or a positive integer multiple of 16 | Planner cap/choice for the named stage. It is bound to the model or policy; construct a new one to change it. |
| `padding_rows` | `"auto"` or a non-negative integer | Exact/automatic execution padding for the named stage. An exact integer is mutually exclusive with `padding_budget`. |
| `padding_budget` | Non-negative integer | Maximum optional padding considered by the stage planner. It cannot accompany an exact integer `padding_rows`. |

All three stage mappings are cold even when a field influences only one Graph:
pass them to the loader/factory before RPU weight installation and first Graph
BUILD. The adapter may narrow a request to its exact capability envelope and
will expose the resolved execution plan where its public policy API documents
one.

### Memory and coherency controls

- `torch.rpu.set_caching_allocator(bool)` selects the process tensor allocator
  (default direct/off). The first explicit selection or non-empty RPU tensor
  allocation freezes the choice; repeating the same value is idempotent and a
  conflicting value raises. Use a fresh process to choose another mode.
  `torch.rpu.empty_cache()` releases cached, unused blocks; it cannot release
  live tensors or Graph-owned storage. Adapters that retain many tensors select
  the common caching policy before their first RPU allocation.
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
- `torch.rpu.set_ddr_flush(bool)` controls internal RPU-to-RPU flush points
  (default off). Model adapters may set it when their cross-component contract
  requires it.
- `torch.rpu.set_ddr_flush_force(bool)` controls CPU/RPU boundary coherency
  (default on). Keep it enabled for normal inference; disabling it is unsafe
  outside an isolated microbenchmark. The matching `get_*` functions report
  current process state.

### Profiling mechanisms

| Mechanism | What it measures | When to enable | Output/risk |
|---|---|---|---|
| `torch.rpu.set_profile(True)` | Backend wrapper and accumulated timing counters | After warmup if only steady-state counters are wanted; call `reset_profile_accumulators()` before the measured window | Lightweight console/counter diagnostics; it is not a PyTorch operation trace. |
| `torch.profiler.profile(...)` | CPU and `PrivateUse1` operation scopes, including Graph capture/replay ranges | Wrap only the calls to measure; include warmup separately according to the benchmark protocol | Produces an operation-level trace/table and adds profiler overhead. |

## Inventory maintenance

The tables contain one row for each current concrete environment/configuration
name. A new reader or facade preset must add a row in the correct tier; removing
the final reader/preset requires removing its row. Automated documentation
checks should compare the row set bidirectionally with source readers and keep
diagnostic variables below the stable `Diagnostic-only inventory` heading.
