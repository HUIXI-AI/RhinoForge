# Runtime configuration

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

## General and cache settings

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_KERNEL_LIB_PATH` | Combined operator asset beside the extension; existing file path | First asset access / **IMPORT** | Process-wide asset selection. A missing, incompatible, or untrusted file prevents safe execution. |
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
| `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED` | `0`; **E1** | Preflight / **MODEL** | Qwen3.5 Vision 2B/4B controlled evaluation; known numerical gates remain in force. |
| `QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED` | `0`; **E1** | Preflight / **MODEL** | Exact Qwen3-VL 32B W8A16 graph-blocked evaluation only. |
| `RPU_WALL_OSS_RTC_ALLOW_NUMERIC_BLOCKED` | `0`; **E1** | Preflight / **MODEL** | Named Wall-OSS RTC profile evaluation; not robot or numerical certification. |
| `RPU_LINGBOT2_ALLOW_UNVALIDATED` | `0`; **E1** | Preflight / **MODEL** | LingBot-VLA-V2 controlled profiles; output is not robot certified. |
| `RPU_INTERNVLA_N1_ALLOW_NUMERIC_BLOCKED` | `0`; **E1** | Preflight / **MODEL** | InternVLA-N1 legacy controlled evaluation. |
| `RPU_INTERNVLA_N1_EXACT_TWOSTAGE` | `0`; **E1** | Preflight / **MODEL** | Exact InternVLA-N1 two-stage profile; also requires the exact environment and a caller-pinned asset manifest. |
| `RPU_S2_SDPA_BF16` | Unset/empty/`0` only; literal `1` and all other values are rejected | Preflight / **MODEL** | InternVLA policy tripwire for an unavailable asset; leave off. |

## Shared advanced execution selectors

These are profile-owned. A facade may set a validated value before model
construction; ambient overrides can break graph, memory, or numerical
contracts.

### Collective execution

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_ALLREDUCE_CHUNK_BALANCED` | Off; **N1** | First native use / **NATIVE** | Balances row chunks for an opted-in collective path. It changes reduction boundaries and needs numeric validation. |
| `RPU_ALLREDUCE_CHUNK_V2` | Off; **N1** | Graph emission / **BUILD** | Keeps an admitted collective path by chunking large rows. Enable only in a profile that owns the relevant shapes. |
| `RPU_ALLREDUCE_FUSED` | Off; **N1** | First native use / **NATIVE** | Selects a fused reduction/residual route with a different accumulation order; numerical validation is mandatory. |
| `RPU_ALLREDUCE_PARTIAL_TWOSTAGE` | Off; **N1** | First native use / **NATIVE** | Extends the selected path to partial producers. Unsafe unless the model profile proves inactive partitions are disposable. |
| `RPU_ALLREDUCE_RING` | Off; `nopace`, `paced`, `profile`, or `1`/`true`/`auto`; other text means off | Graph emission / **BUILD** | Selects a ring schedule. `profile` is only for an exact facade-owned census; wrong selection can change numerics or fail shapes. |
| `RPU_ALLREDUCE_TWOSTAGE` | Off; **N1** | Graph emission / **BUILD** | Selects the two-stage route. Some VLA facades opt in; never enable it globally for an unrelated model. |
| `RPU_ALLREDUCE_V2_CAP` | `409600`; decimal integer, empty or non-positive resets default | First native use / **NATIVE** | Maximum admitted element count for the selected route. Raising it beyond a verified envelope risks incorrect output. |

### Graph and replay

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_DEEP_FAST_REPLAY` | Off; any non-empty value whose first character is not `0` enables | Fused-handle construction / **MODEL** | Skips audited setup work on full-body replay. A profile must prove every skipped input/layout remains valid. |
| `RPU_FASTREPLAY_SKIP_SYNC` | Off; non-empty value not starting with `0` enables | Graph BUILD / **BUILD** | Skips a redundant mutable-parameter scan only for a fully skipped replay. Wrong use can replay stale parameters. |
| `RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN` | Off; exact `1`, `true`, `True`, or `on` enables | First native coexistence use / **NATIVE** | Retains a persistent SPM generation across profile-owned subsystem handoff. Wrong ownership can corrupt later execution. |
| `RPU_GRAPH_DEFER_TO_COPY` | No independent default; legacy alias accepts `auto`/empty, `0`/`off`/`false`, or any other non-empty value for forced on | First host-op gate use / **NATIVE** | Consulted only when `RPU_GRAPH_HOST_OP_DEFER_GATE` is unset. Avoid setting both. |
| `RPU_GRAPH_HOST_OP_DEFER_GATE` | `auto`; `auto`/empty, `0`/`off`/`false`, or any other non-empty value for forced on | First host-op gate use / **NATIVE** | Controls deferral of stable host inputs during capture. Forced-on with unstable storage risks stale data. |
| `RPU_SKIP_IDLE_RECORD_FUNCTION` | Off; `1`/`on`/`true`/`True` enable | First Python graph scope / **MODEL** | Omits idle profiler scopes only while the profiler is off. It preserves scopes when profiling and otherwise reduces host overhead. |

### KV, linear, normalization, and scheduling

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_ADARMS_FUSED_BCAST` | Off; non-empty except `0`/`false`/`False` | Handle/Graph build / **BUILD** | Fuses profile-owned AdaRMS broadcasts. The owning layout must be contiguous and stable. |
| `RPU_KVINSERT_HYBRID3_V16` | Off; non-empty except `0`/`false`/`False` | Each KV insertion / **CALL**; rebuild Graph | Selects a three-part aligned/tail schedule. Shape and cache signatures must be revalidated. |
| `RPU_KVINSERT_HYBRID_V16` | Off; non-empty except `0`/`false`/`False` | Each KV insertion / **CALL**; rebuild Graph | Selects an aligned bulk/tail schedule. Facades may own it; arbitrary enablement changes graph topology. |
| `RPU_KVINSERT_V16` | On; exact `0` or `false` disables, all other values enable | First native use / **NATIVE** | Selects the aligned KV-insert route when admitted. Disabling changes scheduling and performance, not logical cache length. |
| `RPU_KVINSERT_V16_ANY_TP` | Off; non-empty except `0`/`false`/`False` | First native use / **NATIVE** | Allows the aligned route for additional whole-head parallel factors. Unsupported geometry can fail admission. |
| `RPU_LINEAR_ACC32` | Off; strict `0`/`1`, `false`/`true`, or `off`/`on` with listed case variants; invalid fails | First native use / **NATIVE** | Chooses FP32 accumulation for tiled linear families. It changes tiles, memory use, performance, and numerics. |
| `RPU_LINEAR_AUTOTILE` | Off; exact `1` where checked | Facade admission/model build / **MODEL** | Compatibility marker required by sealed grouped profiles. It is not a general operator-selection API. |
| `RPU_LINEAR_TILE_FORCE` | Auto (`-1`); decimal integer, positive value forces a tile | First native tile selection / **NATIVE** | Overrides every FP16 tiled-linear shape. Invalid tiles fail or mis-size work; restricted tuning only. |
| `RPU_LINEAR_TILE_GY1` | Off; **N1** | First native tile selection / **NATIVE** | Prefers a single-row-group tile where admitted. Changes performance and graph census. |
| `RPU_LINEAR_TILE_M384N` | Auto (`-1`); positive decimal tile | First native tile selection / **NATIVE** | FP16 narrow-output override for rows up to 384; restricted profile tuning. |
| `RPU_LINEAR_TILE_M384W` | Auto (`-1`); positive decimal tile | First native tile selection / **NATIVE** | FP16 wide-output override for rows up to 384; restricted profile tuning. |
| `RPU_LINEAR_TILE_M64` | Auto (`-1`); positive decimal tile | First native tile selection / **NATIVE** | FP16 override for rows up to 64; restricted profile tuning. |
| `RPU_LINEAR_TILE_VIS` | Auto (`-1`); positive decimal tile | First native tile selection / **NATIVE** | FP16 large/vision-shape override; restricted profile tuning. |
| `RPU_LINEAR_TILE_W8_FORCE` | Auto (`-1`); positive decimal tile | First native tile selection / **NATIVE** | Overrides every W8A16 tiled-linear shape; checkpoint/profile validation is required. |
| `RPU_LINEAR_TILE_W8_M384W` | Auto (`-1`); positive decimal tile | First native tile selection / **NATIVE** | W8A16 wide-output override for rows up to 384; restricted profile tuning. |
| `RPU_LINEAR_TILE_W8_M64` | Auto (`-1`); positive decimal tile | First native tile selection / **NATIVE** | W8A16 override for rows up to 64; restricted profile tuning. |
| `RPU_LINEAR_TILE_W8_M64N` | Auto (`-1`); positive decimal tile | First native tile selection / **NATIVE** | W8A16 narrow-output override for rows up to 64; restricted profile tuning. |
| `RPU_LINEAR_TILE_W8_VIS` | Auto (`-1`); positive decimal tile | First native tile selection / **NATIVE** | W8A16 large/vision-shape override; restricted profile tuning. |
| `RPU_RMSNORM_NEWTON` | Off; strict `0`/`1`, `false`/`true`, or `off`/`on` with listed case variants; invalid fails | First native use / **NATIVE** | Selects refined normalization. It changes numerics and can change available schedules. |
| `RPU_RMSNORM_VWARP` | `0`; `0`, `16`, `32`, or `auto`; unknown text falls back to `0` | First native use / **NATIVE** | Selects an admitted vector row schedule. Unsupported divisibility falls back; timing changes. |

### Qwen3.5 Vision selectors

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `QWEN3_5_VISION_CHUNK` | Auto; decimal integer, empty/non-positive means auto | First native vision plan / **NATIVE** | Caps Qwen3.5 Vision chunking. Positive overrides must remain in the admitted shape and memory envelope. |
| `RPU_VISION_MERGER` | **PB(true)** | Qwen3.5 Vision setup / **MODEL** | Enables the RPU merger path. `0` selects a fallback and changes performance/numerics. |
| `RPU_VISION_STEP0` | **PB(true)** | Qwen3.5 Vision setup / **MODEL** | Enables the RPU first-stage path. `0` selects a fallback and changes performance/numerics. |

## GR00T profile

The public GR00T builder supplies its facade defaults before construction.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_GR00T_KVPAD16` | Raw off; facade `1`; native `1` or a value beginning with `t`/`T` enables | First native KV setup / **NATIVE** | Pads and masks KV rows for the GR00T expert. It changes cache layout and Graph signatures. |
| `RPU_GR00T_PARTIAL_MROPE` | Raw **PB(false)**; facade **PB(true)** | GR00T construction / **MODEL** | Uses precomputed partial multimodal position tables. The exact SPM-Z1 profile instead requires it off. |
| `RPU_GR00T_QWEN3VL_SPM_Z1` | Off; exact `1` or lowercase `true` | GR00T construction / **MODEL** | Enables the exact FP16 physical Vision/Text handoff profile and validates every companion selector. Other shapes/precisions fail. |
| `RPU_GR00T_W8A16` | **PB(false)** | GR00T weight installation / **MODEL** | Quantizes the admitted backbone and action projections. It conflicts with the exact FP16 SPM-Z1 profile and changes numerics. |

## Hy-Embodied profile

`HyEmbodiedPolicy` applies a complete scoped profile before construction. The
raw defaults below describe direct adapter use; the facade defaults named in a
row are the normal public-policy values. Use a fresh process to switch policy
profiles.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_HY_VLA_ACTION_MLP_MC` | Raw unset means 8 cores; `0`/`off`/`false`/`False` means 1, all other values mean 8; facade `1` | Weight/build setup / **MODEL** | Selects multi-core action-MLP execution. It changes reduction order and must retain profile numerical gates. |
| `RPU_HY_VLA_ATTN_TP8` | Raw off; `1`/`true`/`True`/`on`; facade `1` | Weight installation / **MODEL** | Replicates KV heads for eight-core attention. Cache geometry and weights must be rebuilt. |
| `RPU_HY_VLA_CACHING_ALLOC` | On; `0`/`false`/`False`/`off` disables | Hy-VLA construction / **MODEL** | Enables the process caching allocator for this policy. It changes memory retention; call `empty_cache()` before measuring free memory. |
| `RPU_HY_VLA_DENOISE_UNROLL` | Raw off; `1`/`true`/`True`/`on`; facade `1` | Action construction / **MODEL** | Records the denoise loop as one profile-owned Graph. It changes Euler-state precision and numerical results. |
| `RPU_HY_VLA_FAST_REPLAY` | Raw off; `0`/`off`/`false`, `1`/`on`/`true`, or a value naming `vit`, `vlm`, and/or `expert`; facade `1` | First native subsystem use / **NATIVE** | Skips audited layer-body emission on replay. Wrong use can leave dynamic inputs stale. |
| `RPU_HY_VLA_FAST_REPLAY_PRELOAD` | Raw off; `0`/`off`/`false`, `1`/`on`/`true`, or a value naming `vit`, `vlm`, and/or `expert`; facade `1` | First native subsystem use / **NATIVE** | Also skips audited preload work on replay. It requires stable persistent state. |
| `RPU_HY_VLA_FUSED_MERGER` | Raw off; Python accepts `1`/`true`/`True`/`on`; native mirror requires literal `0`/`1`; facade `1` | Vision construction / **MODEL** | Folds the vision merger into the graph. Parser mismatch is prevented by using only `0`/`1`. |
| `RPU_HY_VLA_KVPAD16` | Raw off; `0`/`off`/`false`, `1`/`on`/`true`, or a value naming `vit`, `vlm`, and/or `expert`; facade `1` | First native subsystem use / **NATIVE** | Pads/masks expert KV rows. It changes cache layout and signatures. |
| `RPU_HY_VLA_MASK_ONCE` | Raw off; `0`/`off`/`false`, `1`/`on`/`true`, or a value naming `vit`, `vlm`, and/or `expert`; facade `1` | First native subsystem use / **NATIVE** | Reuses an invariant mask upload. It is unsafe when mask contents or storage can change. |
| `RPU_HY_VLA_MERGER_IN_GRAPH` | Raw off; `1`/`true`/`True`/`on`; facade `1` | Vision construction / **MODEL** | Places the enabled fused merger inside capture. Requires `RPU_HY_VLA_FUSED_MERGER=1`. |
| `RPU_HY_VLA_MOT_NORM_NOMERGE` | `both`; `0`/`off`/`false`, `1`/`on`/`true`/`both`, or a value containing `qkv` and/or `mlp` | First native VLM use / **NATIVE** | Selects profile-specific norm/merge scheduling. It changes temporary-memory and graph structure. |
| `RPU_HY_VLA_PARTIAL_ROPE` | Raw off; `0`/`off`/`false`, `1`/`on`/`true`, or CSV scopes `vit`, `vlm`, `expert`; facade `expert,vlm` | First native subsystem use / **NATIVE** | Uses precomputed partial position tables for selected subsystems. Scope mismatch changes position semantics. |
| `RPU_HY_VLA_PATCH_EMBED_IN_GRAPH` | On; `0`/`false`/`False`/`off` disables; facade `1` | Vision construction / **MODEL** | Captures device patch embedding. Disabling selects a different host/device boundary. |
| `RPU_HY_VLA_PATCH_EMBED_MC` | Raw unset means 8 cores; `0`/`false`/`False` means 1; facade `1` | Weight and vision construction / **MODEL** | Selects multi-core patch embedding. Python/native state must agree; rebuild weights and Graph. |
| `RPU_HY_VLA_PERSIST_HANDLES` | Raw off; `0`/`off`/`false`/empty, `1`/`on`/`true` for all, or CSV subset `vit`, `vlm`, `expert`; facade `1` | Policy construction / **MODEL** | Keeps selected handles alive across calls. It increases retained memory and enforces one-policy ownership. |
| `RPU_HY_VLA_PREFIX_TEMPLATE` | On; `0`/`false`/`False`/`off` disables | Prompt construction / **MODEL** | Enables the profile prompt template. Changing it changes token input, not merely performance. |
| `RPU_HY_VLA_PROJ1_IN_MERGER` | On; `0`/`off`/`false`/`False` disables; native mirror uses the same variable | Vision construction / **MODEL** | Includes the first projection in the merger path. Use literal `0`/`1`; disagreement breaks output shape/ownership. |
| `RPU_HY_VLA_Q_INPLACE` | Raw off; non-empty except `0`/`off`/`false`; facade `1` | First native vision use / **NATIVE** | Reuses the query buffer in place. It is safe only when the profile proves the prior value dead. |
| `RPU_HY_VLA_RMSNORM_PAD16` | Raw off; `0`/`off`/`false`, `1`/`on`/`true`, or a value naming `vit`, `vlm`, and/or `expert` | First native subsystem use / **NATIVE** | Pads selected normalization rows. It changes layout and Graph census. |
| `RPU_HY_VLA_SILU_MUL` | Raw off; `0`/`off`/`false`, `1`/`on`/`true`, or CSV scopes `vit`, `vlm`, `expert`; facade `expert` | First native subsystem use / **NATIVE** | Fuses the selected activation/multiply schedule. It needs per-scope numerical validation. |
| `RPU_HY_VLA_VIT_PACKED` | Raw off; `1`/`true`/`True`/`on`; facade `1` | Vision construction / **MODEL** | Packs the vision-tower execution profile. It changes shape/layout contracts and requires rebuilt weights/Graph. |
| `RPU_HY_VLA_W4A16` | Off; `0`/`off`/`false`/`False`/empty, `1`/`on`/`true`/`True`/`all`, or CSV subset `vit`, `vlm`, `expert`, `vlm_text`, `vlm_vision` | Weight installation / **MODEL** | Selects W4A16 per scope; W4 takes precedence over W8 on overlap and unsupported scopes fail. Numeric profile changes. |
| `RPU_HY_VLA_W8A16` | Raw off; facade maps `fp16`→`0`, `w8a16`→`all`, `w8a16-expert`→`expert`, `w8a16-expert-vlmv`→`expert,vlm_vision`, `w8a16-vlm`→`vlm`, `w8a16-vit`→`vit`, `w8a16-no-vlm`→`vit,expert`; also accepts the W4 row's exact enum/CSV grammar | Weight installation / **MODEL** | Selects W8A16 per scope. It must match checkpoint/source weights and the named precision profile. |

## InternVLA-N1 and NavDP profiles

The controlled InternVLA path also requires a caller-supplied SHA256-pinned
asset manifest through its Python API; no environment path replaces it.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_INTERNVLA_DINO_FINAL_NORM_RPU` | **PB(true)** | DINO tower construction / **MODEL** | Runs the final normalization on RPU. `0` changes the host/device boundary and timing. |
| `RPU_INTERNVLA_DINO_NORM_FOLD` | **PB(false)** | DINO tower construction / **MODEL** | Folds input normalization into patch weights. It changes installed weights and must be rebuilt. |
| `RPU_INTERNVLA_HOST_CACHE` | **PB(true)** | Backbone construction / **MODEL** | Enables host memoization and stable owners. Disabling increases repeated host work. |
| `RPU_INTERNVLA_NEXTDIT_FOLD_NORM_MOD` | **B01(false)** | NextDiT construction / **MODEL** | Folds normalization/modulation work into the installed profile. Changes weights/layout and numerics. |
| `RPU_INTERNVLA_NEXTDIT_FUSED_QKQ` | **PB(false)** | NextDiT construction / **MODEL** | Enables the fused Q/K/query-preparation path. It changes Graph structure and numerical ordering. |
| `RPU_INTERNVLA_NEXTDIT_STEP_PREP_RPU` | **PB(false)** | NextDiT construction / **MODEL** | Moves per-step preparation to RPU. It changes graph inputs and host/device boundaries. |
| `RPU_NAVDP_DENOISE_UNROLL` | Off; stripped `1`/`true`/`True` enables, other values disable | NavDP action call / **CALL**; rebuild Graph | Captures the fixed denoise loop. It changes Graph topology and step execution. |
| `RPU_NAVDP_BATCH` | On; stripped `1`/`true`/`True` enables, other values disable | First unrolled action dispatch / **CALL**; rebuild Graph/model | Batches trajectories. When on, sample count is fixed after first dispatch; later mismatches fail. |

## LingBot-VLA-V2 profile

Prefer `Lingbot2Policy.from_checkpoint(..., runtime_env=...)`. The facade
validates an exact allowlist, applies a full snapshot before adapter import, and
restores the shell environment on close; native state is still process-cold, so
use a fresh process for another profile.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE` | Raw/facade `0`; `1`/`true`/`True`/`on` | Expert construction / **MODEL** | Uses a direct indexed AdaRMS schedule. It conflicts with denoise unroll and requires expert replay. |
| `RPU_LINGBOT2_BASE_W8A16` | Raw `0`; **E1**; W8/W4 facades `1` | Weight conversion / **MODEL** | Quantizes shared/base expert projections. It is part of the complete precision profile, not an independent toggle. |
| `RPU_LINGBOT2_DENOISE_TWOSTAGE` | Raw `0`; **E1**; exact grouped Z2 profiles `1` | Expert handle build / **MODEL** | Enables the product-scoped denoise collective route. Other profiles must leave it off. |
| `RPU_LINGBOT2_DENOISE_UNROLL` | Raw `0`; facade `1`; facade normalizes only exact `1` to on | Policy construction / **MODEL** | Records the fixed ten-step denoise Graph and uses FP16 Euler state. `0` restores the host loop and changes numerics. |
| `RPU_LINGBOT2_DOWN_ACC32_TILE` | Raw `0`; **E1**; ordinary FP16 facade `1` | First linear use / **NATIVE** | Selects the grouped down-projection ACC32 tile. Precision/profile mismatch changes numerics or fails. |
| `RPU_LINGBOT2_ENCODER_1THREAD` | `1`; `1`/`true`/`True`/`on` enables | Policy construction / **MODEL** | Runs small host encoder work with one thread. `0` uses the caller's process-wide thread setting. |
| `RPU_LINGBOT2_EXPERT_REPLAY` | Facade `1`; `1`/`true`/`True`/`on` | Policy construction / **MODEL** | Enables stable expert Graph replay. It requires stable input owners; disabling increases build/dispatch work. |
| `RPU_LINGBOT2_EXPERT_W4A16` | Raw `0`; **E1**; W4 facade `1` | Weight conversion / **MODEL** | Selects W4A16 routed-expert weights. It wins over W8 if both are set; use only the complete W4 profile. |
| `RPU_LINGBOT2_EXPERT_W8A16` | Raw `0`; **E1**; W8 facade `1` | Weight conversion / **MODEL** | Selects W8A16 routed-expert weights unless W4 is on. Numeric profile changes. |
| `RPU_LINGBOT2_FP16_TOP4` | Native default on unless dense-router diagnostic is enabled; facade `1`; exact `0`/`1` | Expert weight installation / **MODEL** | Keeps strict FP16 top-4 routing. Disabling without the dense diagnostic fails; routing changes are accuracy-sensitive. |
| `RPU_LINGBOT2_GROUPED_EXPERTS` | Raw `0`; **E1**; ordinary W8/W4/FP16 facades `1`, dense Z2 `0` | Weight conversion and handle build / **MODEL** | Selects packed grouped experts. It must agree with weight format and row-chunk profile. |
| `RPU_LINGBOT2_HOST_PREFIX_OPT` | Facade `1`; `1`/`true`/`True`/`on` | Policy construction / **MODEL** | Keeps stable prompt-prefix owners and memoization. Required for VLM replay and direct-prefix output. |
| `RPU_LINGBOT2_LEGACY_PREPROC` | `0`; **E1** | Policy construction / **MODEL** | Compatibility override that forces legacy preprocessing. It changes model input and is not a performance-only switch. |
| `RPU_LINGBOT2_MULTIVIEW_SPM_Z2` | `0`; **E1**, accepted only through exact policy `runtime_env` | Policy construction / **MODEL** | Enables the sealed three-camera physical profile with exact shapes, precision, capacities, and companion selectors; all drift fails. |
| `RPU_LINGBOT2_OUTER_TWOSTAGE` | Raw `0`; exact `0`/`1`; grouped Z2 `1` | Multiview profile preflight / **MODEL** | Selects the outer product-scoped collective route. It has no meaning outside the sealed Z2 profile. |
| `RPU_LINGBOT2_PREFILL_FAST_REPLAY` | Raw `0`; `1`/`true`/`True`/`on`; ordinary facade `1`, Z2 `0` | Prefill handle setup / **MODEL** | Enables audited prefill body replay skips. Requires VLM replay and stable prefix storage. |
| `RPU_LINGBOT2_PREFILL_W4A16` | Raw/facade `0`; `1`/`true`/`True`/`on` | Weight conversion / **MODEL** | Restricted option quantizing only admitted prefill projections to W4A16. Ordinary W4 uses prefill W8A16 instead. |
| `RPU_LINGBOT2_PREFILL_W8A16` | Raw `0`; `1`/`true`/`True`/`on`; W8/W4 facades `1` | Weight conversion / **MODEL** | Selects W8A16 prefill projections. It must be part of the complete precision profile. |
| `RPU_LINGBOT2_PREPROC` | `exact`; lower-case `exact`, `fast`, or `legacy`; other facade input normalizes to `exact` before build | Policy construction / **MODEL** | Selects preprocessing semantics. It changes model input; exact Z2 requires `exact`. |
| `RPU_LINGBOT2_QWEN3VL_BASE` | Packaged/registry fallback; non-empty local directory containing a valid config | Policy construction / **MODEL** | Overrides the Qwen3-VL base assets. Wrong or untrusted contents fail validation; it is not a download credential. |
| `RPU_LINGBOT2_VISION_DIRECT_PREFIX` | Facade `0`; `1`/`true`/`True`/`on` | Policy construction / **MODEL** | Writes fused vision results into stable prefix owners. Requires host-prefix optimization, VLM replay, batching, and fused merger. |
| `RPU_LINGBOT2_VISION_DIRECT_PREFIX_NO_OUTPUT` | Facade `0`; `1`/`true`/`True`/`on` | Policy construction / **MODEL** | Suppresses the extra vision return only with direct-prefix enabled. Wrong pairing fails. |
| `RPU_LINGBOT2_VISION_FAST_REPLAY` | Raw `0`; `1`/`true`/`True`/`on`; ordinary facade `1`, Z2 `0` | Vision handle setup / **MODEL** | Enables audited vision replay skips. It requires the named vision topology and stable dynamic inputs. |
| `RPU_LINGBOT2_VISION_PACKED_PREP` | Facade preset `1`; no retained standalone environment reader | Facade profile composition / **MODEL** | Compatibility marker for the sealed multiview preparation bundle. Do not use it as an independent public-runner override. |
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

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_QWEN3VL_VISION_BATCH` | **PB(false)** | Vision forward / **CALL**; rebuild Graph | Packs compatible images/views. It changes Graph signatures and memory; facade constraints still cap shapes. |
| `RPU_QWEN3VL_VISION_BATCH_CAP` | `3`; decimal integer clamped to at least `1` | Vision forward / **CALL**; rebuild Graph | Maximum compatible images per packed group. Larger values can exceed the profile memory envelope. |
| `RPU_QWEN3VL_VISION_FUSED_MERGER` | Raw off; **B01(false)** | Vision construction / **MODEL** | Folds the merger into the Vision Graph. Only literal `0`/`1` keeps Python and native readers consistent. |
| `RPU_QWEN3VL_VISION_HOST_FP32_PATCH` | `0`; **E1** | Vision forward / **CALL**; rebuild Graph/model | Uses host FP32 patch projection instead of the default FP16/device profile. It changes numerics and transfer cost. |
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

## Wall-OSS profile

`WallOssPolicy` supplies validated facade defaults before model construction.
Prompt selectors change the actual model input and must never be treated as
performance-only switches.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `RPU_WALL_OSS_ACTION_FP32_TAIL` | **PB(false)** | Each denoise call / **CALL**; rebuild prepared Graph profile | Uses host FP32 for the Euler tail. It changes action numerics and conflicts with fused RTC/unroll profiles. |
| `RPU_WALL_OSS_ATTN_TP8` | Raw **PB(false)**; facade `1` | Weight installation / **MODEL** | Replicates KV heads for eight-core attention. Rebuild weights, caches, and Graphs. |
| `RPU_WALL_OSS_BATCH_MAX_SEQ` | `768`; decimal integer | Vision grouping / **CALL**; rebuild Graph | Caps packed vision sequence length. Too large can exceed memory; mixed-precision and exact profiles further constrain it. |
| `RPU_WALL_OSS_BATCH_VISION` | **PB(true)** | Vision forward / **CALL**; rebuild Graph | Packs equal-grid images up to the cap. It changes signatures and memory. |
| `RPU_WALL_OSS_DENOISE_UNROLL` | **PB(true)** | Action construction and calls / **MODEL** | Uses the fused denoise loop. `0` selects the host loop; RTC requires it on. |
| `RPU_WALL_OSS_DEVICE_PATCH_EMBED` | **PB(true)** | Vision construction / **MODEL** | Runs folded patch embedding on RPU. `0` uses CPU projection and changes boundary/performance. |
| `RPU_WALL_OSS_EXPERT0_DOWN_INT4` | **PB(true)** | Quantized LLM weight installation / **MODEL** | Uses the admitted compact expert-0 down projection. Some RTC/W4 profiles force it off; mismatches fail validation. |
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
| `RPU_WALL_OSS_PARTIAL_MROPE` | **PB(true)**; facade sets `1` | LLM construction / **MODEL** | Uses the required Wall-OSS multimodal position-table path. Explicit `0` is rejected for supported Wall-OSS input. |
| `RPU_WALL_OSS_PE_NORM_FOLD` | **PB(true)** | Policy construction / **MODEL** | Folds image normalization into patch weights when fast preprocessing is active. Rebuild installed weights to change it. |
| `RPU_WALL_OSS_PREFILL_MIXED_PRECISION` | Raw **B01(false)**; RTC facade `1` | Decoder weight/handle installation / **MODEL** | Enables the fixed overflow-safe prefill profile. It changes stored intermediate precision and numerics. |
| `RPU_WALL_OSS_PREFILL_STAGED_MLP` | **B01(false)** | Decoder handle installation / **MODEL** | Enables the restricted staged prefill MLP and requires the matching mixed-precision/checkpoint profile. |
| `RPU_WALL_OSS_RTC_EMBODIMENT` | `X2 ex001_desktop` in RTC; non-empty string | RTC prompt construction / **CALL** | Selects the embodiment text in the RTC prompt. Empty fails; any change alters model input and actions. |
| `RPU_WALL_OSS_SHORT_PROMPT` | **PB(false)** | Prompt construction / **CALL** | Selects the short prompt schema; mutually exclusive with other prologue modes. Changes tokens. |
| `RPU_WALL_OSS_VISION_FUSED_MERGER` | Raw **B01(false)**; facade `1` | Vision construction / **MODEL** | Folds the merger into the Vision Graph. Only literal `0`/`1` keeps Python/native state consistent. |
| `RPU_WALL_OSS_VISION_LAYER_GROUP` | `0`; exact `0` or `32` | Vision construction (**MODEL**) and first native schedule claim (**NATIVE**); fresh process | Restricted exact three-image scheduling experiment; mutually exclusive with weight-outer and several mixed profiles. |
| `RPU_WALL_OSS_VISION_MIXED_PRECISION` | **B01(false)**; RTC facade `1` | Vision construction / **MODEL** | Enables the fixed overflow-safe vision profile. It requires fused merger and exact supported shapes/checkpoints. |
| `RPU_WALL_OSS_VISION_ROPE_SPM` | Raw off; **N1**; facade `1` | First native vision use / **NATIVE** | Keeps position tables in SPM. It changes persistent-memory use and requires a rebuilt vision handle. |
| `RPU_WALL_OSS_VISION_WEIGHT_OUTER` | Raw **B01(false)**; exact RTC W8 may default `1` | Vision construction (**MODEL**) and first native schedule claim (**NATIVE**); fresh process | Compatibility name for the restricted exact three-image schedule. It is mutually exclusive with layer grouping. |

## Diagnostic-only inventory

The comprehensive public runner rejects every variable in this section when it
appears in TOML. A developer may still set one explicitly in the shell for a
controlled local investigation. These controls can change execution, dump model
data, expose addresses or timing structure, or invalidate correctness/performance
claims. Leave them unset in production and review every generated artifact before
sharing it.

| Variable | Unset/default and accepted values | Read / change | Scope, effect, and risk |
|---|---|---|---|
| `HALO_CHUNK_DIAG` | Off by absence; any presence, including empty or `0`, enables | Chunk planning / **CALL** | Prints planner decisions. Adds synchronization/log volume and may expose model shapes. |
| `HALO_NODE_HIST` | Off by absence; any presence, including empty or `0`, enables | Graph segment BUILD / **BUILD** | Prints node/segment histograms. Output exposes execution structure and is not a support gate. |
| `HALO_PROFILE` | Off; exact `1` | HALO action call / **CALL** | Runs `torch.profiler` around the call and writes a trace/table. Profiling overhead invalidates latency results unless intentionally measured. |
| `HALO_PROFILE_DIR` | `output/halo_profile`; directory path | Profile dump / **CALL** | Destination used only with `HALO_PROFILE=1`. Files can contain paths, shapes, and timing structure. |
| `QWEN3_5_VISION_DBG_Q` | Off; exact `1` or lowercase `true` | First native debug check / **NATIVE** | Enables the Qwen3.5 Vision query probe path. It adds captures/synchronization and may expose intermediates. |
| `QWEN3_5_VISION_GRAPH_DISABLE` | `0`; any value other than exact `0` disables | Vision forward / **CALL**; rebuild/clear Graph | Bypasses Qwen3.5 Vision GraphCache for controlled comparison. Replay and latency conclusions no longer apply. |
| `RPU_ALLREDUCE_TWOSTAGE_TRACE` | Off; **N1** | Graph emission / **BUILD** | Prints one routing line per distinct admitted shape/path. Logs expose topology and add output overhead. |
| `RPU_ALL_GATHER_FORCE_MULTI_CORE` | Off; **N1** | First native use / **NATIVE** | Forces the multi-core schedule for A/B comparison. It is not a generally supported performance selector. |
| `RPU_CHUNK_FORCE_UNSAFE` | Off; **N1** | First applicable native planner/launcher use / **NATIVE** | Bypasses a planner safety check. It can exceed execution constraints and must never produce deployable output. |
| `RPU_DYNAMO_MATERIALIZE_BREAKS` | **PB(false)** | Dynamo partitioning / **CALL**; recompile | Materializes partition breaks. It changes graph boundaries and adds transfers, so it is only a compiler diagnostic. |
| `RPU_GRAPH_DDR_SPM_LOG` | Off; non-empty whose first character is not `0` | First data-node execution / **NATIVE** | Logs copy endpoints, byte counts, and checksums. Output is sensitive and logging changes timing. |
| `RPU_GRAPH_FORCE_ONESHOT_ON_REPLAY` | Off; non-empty whose first character is not `0` | First replay check / **NATIVE** | Executes a one-shot route instead of normal replay. It invalidates Graph lifecycle and performance conclusions. |
| `RPU_GRAPH_HCB_CHECKSUM` | Off; non-empty whose first character is not `0` | First host-callback execution / **NATIVE** | Logs live/stable tensor metadata, device addresses, and value summaries. Treat output as sensitive model data. |
| `RPU_GRAPH_TO_COPY_LOG` | Off; non-empty whose first character is not `0` | Capture/replay copy handling / **CALL** | Logs storage/device addresses and copy adoption. Output is sensitive and highly verbose. |
| `RPU_KVINSERT_V16_TRACE` | Off; any present value except exact `0` or `false`; an empty exported value enables | First native use / **NATIVE** | Logs KV route selection and shapes. It is diagnostic output, not proof of cache correctness. |
| `RPU_L2_ADDR` | Off by absence; any presence, including `0`, enables | First grouped-expert Graph emission / **BUILD** | Prints SPM buffer addresses/sizes. Treat logs as sensitive architecture/runtime artifacts. |
| `RPU_L2_BUFONLY` | Off by absence; any presence, including `0`, enables | Grouped-expert Graph emission / **BUILD** | Declares grouped buffers but runs the per-expert route for bisection. Changes graph and performance. |
| `RPU_L2_CAPTURE_GATE` | Off; **E1** | Expert weight installation / **MODEL** | Allocates and exports gate-related layer-0 intermediates. Increases memory/Graph work and exposes model data. |
| `RPU_L2_CAPTURE_INNORM` | Off; **E1** | Expert weight installation / **MODEL** | Captures layer-0 post-normalization input. The tensor may contain user-derived activations. |
| `RPU_L2_CAPTURE_L0` | Off; **E1** | Expert weight installation / **MODEL** | Captures layer-0 output for comparison. Adds persistent storage and a copy. |
| `RPU_L2_CAP_RESID` | Off; **E1** | Expert weight installation / **MODEL** | Captures residual-stream tensors across layers. High memory cost; outputs may contain request-derived activations. |
| `RPU_L2_DBG_PACKED` | Off; **E1** checked at each debug getter | Debug getter / **CALL** | Unlocks access to packed model weights for local inspection. Never expose returned tensors. |
| `RPU_L2_DOWN_ACC16` | Off; **E1** | Expert weight installation / **MODEL** | Forces the diagnostic ACC16 grouped-down route. It changes numerical results and is not a supported precision profile. |
| `RPU_L2_RCHUNK` | Runtime fallback `256`; positive decimal; supported grouped profiles require exact `1632` | Weight/profile validation (**MODEL**) and Graph emission (**BUILD**); fresh process | Overrides grouped scaling row chunks. Wrong values fail the sealed profile or change graph/timing. |
| `RPU_L2_ROUTED_ONLY` | Off; **E1** | Expert weight installation / **MODEL** | Isolates routed-expert contribution for bisection. Output is not full-model output. |
| `RPU_L2_SCHUNK` | `32512`; positive decimal, non-positive uses default | Graph emission / **BUILD** | Overrides grouped activation host chunking. It changes graph census and may reduce safety/performance. |
| `RPU_L2_STAGES` | Off; **E1** | Expert weight installation / **MODEL** | Captures intermediate grouped stages. Adds memory/copies and exposes activations. |
| `RPU_LINGBOT2_DEBUG_DENSE_SOFT_ROUTER` | Off; **E1** | Expert weight installation / **MODEL** | Replaces strict top-4 routing with dense soft routing. Accuracy is unvalidated and output is non-production. |
| `RPU_LINGBOT2_DEBUG_DUMP_ROUTER_H` | Off; **E1** | Expert weight installation / **MODEL** | Captures per-layer router inputs. Dumps can contain request-derived activations and use substantial memory. |
| `RPU_P7_SPLIT_KERNELS` | Empty; comma-separated, whitespace-trimmed name substrings | First segment planning / **NATIVE** | Forces matching operations into isolated segments for bisection. Changes segmentation and invalidates timing. |
| `RPU_PI05_LOAD_NOISE` | Unset; path to an existing tensor `.pt`; missing path is ignored | Noise preparation / **CALL** | Replaces sampled noise after strict shape/finite checks using `weights_only=True`. Only trusted local files should be used. |
| `RPU_PI05_LOAD_PIXEL_VALUES` | Unset; path to an existing tensor `.pt`; missing path is ignored | Image feature call / **CALL** | Replaces processed pixel values after strict checks. It changes model input and may load sensitive test data. |
| `RPU_PI05_LOAD_PREFIX_EMBS` | Unset; path to an existing tensor `.pt`; missing path is ignored | Prefix preparation / **CALL** | Replaces prefix embeddings after strict checks. It bypasses normal upstream values and invalidates E2E claims. |
| `RPU_PI05_LOG_CONVERSION` | Off by absence; any non-empty value, including `0`, enables | Gemma prefill / **CALL** | Logs conversion/handle/chunk diagnostics. Adds output and may expose model shape/configuration. |
| `RPU_PI05_PROBE_DIR` | Unset; non-empty directory path | Pi0.5 conversion and forwards / **CALL** | Dumps named inputs, activations, and KV tensors as `.pt`. Artifacts can contain model weights and user data. |
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
three possible stages are `prefill`, `vision`, and `action`; each entry point
advertises the subset it supports and rejects every other stage or field before
weight loading.

| Field | Accepted value | Meaning and lifecycle |
|---|---|---|
| `chunk_size` | `"auto"` or a positive integer multiple of 16 | Planner cap/choice for the named stage. It is bound to the model or policy; construct a new one to change it. |
| `padding_rows` | `"auto"` or a non-negative integer | Exact/automatic execution padding for the named stage. An exact integer is mutually exclusive with `padding_budget`. |
| `padding_budget` | Non-negative integer | Maximum optional padding considered by the stage planner. It cannot accompany an exact integer `padding_rows`. |

All three stage mappings are cold even when a field influences only one Graph:
pass them to the loader/factory before RPU weight installation and first Graph
BUILD. The adapter may narrow a request to its exact capability envelope and
will expose the resolved execution plan where its public policy API documents
one.

### Memory and coherency controls

- `torch.rpu.set_caching_allocator(bool)` toggles the process caching allocator
  (default off). `torch.rpu.empty_cache()` releases cached, unused blocks; it
  cannot release live tensors or Graph-owned storage.
- `torch.rpu.memory_stats()` and `torch.rpu.get_memory_stats()` return allocator
  counters. `reset_peak_memory_stats()` resets peak counters, while
  `reset_accumulated_memory_stats()` resets accumulated allocation/free
  counters. Resetting counters does not free memory.
- `torch.rpu.set_ddr_flush(bool)` controls internal RPU-to-RPU flush points
  (default off). Model adapters may set it when their cross-component contract
  requires it.
- `torch.rpu.set_ddr_flush_force(bool)` controls CPU/RPU boundary coherency
  (default on). Keep it enabled for normal inference; disabling it is unsafe
  outside an isolated microbenchmark. The matching `get_*` functions report
  current process state.

### Three profiling mechanisms

| Mechanism | What it measures | When to enable | Output/risk |
|---|---|---|---|
| `torch.rpu.set_profile(True)` | Backend wrapper and accumulated timing counters | After warmup if only steady-state counters are wanted; call `reset_profile_accumulators()` before the measured window | Lightweight console/counter diagnostics; it is not a PyTorch operation trace or hardware trace. |
| `torch.profiler.profile(...)` | CPU and `PrivateUse1` operation scopes, including Graph capture/replay ranges | Wrap only the calls to measure; include warmup separately according to the benchmark protocol | Produces an operation-level trace/table and adds profiler overhead. |
| `torch.rpu.hw_perf_trace(...)` or `set_hw_perf_trace(True, ...)` | Raw per-Graph hardware timing trace | Enable **before model construction and the first Graph BUILD**. Existing adapter-owned GraphCache entries are not guaranteed to invalidate or rebuild when toggled; recreate/rebuild the model/Graphs instead of relying on lazy invalidation. | Writes bounded trace files that can include device addresses and execution structure. Treat them as sensitive, summarize locally, and remove raw files before sharing. |

The `hw_perf_trace` context manager disables tracing on normal or exceptional
exit. Enter the context before calling the model factory, not merely around a
warm replay.

## Inventory maintenance

The tables contain one row for each current concrete environment/configuration
name. A new reader or facade preset must add a row in the correct tier; removing
the final reader/preset requires removing its row. Automated documentation
checks should compare the row set bidirectionally with source readers and keep
diagnostic variables below the stable `Diagnostic-only inventory` heading.
