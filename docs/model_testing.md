# Model execution and profiling

[简体中文](model_testing.zh.md) | English

RhinoForge provides two manual execution layers:

- the scripts in `examples/` are the shortest readable model entry points;
- `examples/run_model.py` runs one of those same scripts while applying a
  documented runtime environment and optional Torch and RPU hardware profiles
  from TOML.

These are board smoke and diagnostic entry points. A successful run proves
that the selected local profile executes; it does not replace a CPU/golden
numerical comparison or expand the support status in
[Model support](model_support.md).

## Entry-point inventory

| Model/profile | Direct entry | Full-runner target | Availability |
|---|---|---|---|
| Qwen3 FP16, Llama-3.2-1B | `examples/causal_lm.py` | `causal_lm` | Runnable with a matching exact release checkpoint/profile |
| Qwen3 14B W8A16 | `examples/causal_lm.py` | `causal_lm` | Source-only local-conversion entry; v1.0.0 binds no public immutable derived checkpoint identity or hash |
| Qwen3.5 text | `examples/qwen3_5_text.py` | `qwen3_5_text` | Runnable text entry |
| Qwen3.5 Vision 2B/4B | `examples/qwen3_5_vision.py` | `qwen3_5_vision` | Experimental/numeric-blocked image entry; exact `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1` is required for controlled evaluation and is not a support claim |
| Qwen3-VL 2B/4B | `examples/qwen3_vl.py` | `qwen3_vl` | Runnable image-and-text entry |
| Qwen3-VL 8B dense FP16 | `examples/qwen3_vl.py` | `qwen3_vl` | Limited exact padded-profile entry; 4/4 numeric/task/Graph legs pass, including dual-full, with base 5/5 and long-text 9/9 tokens exact; nearby lengths, video, and quantized paths do not inherit |
| Qwen3-VL 32B W8A16 | `examples/qwen3_vl.py` | `qwen3_vl` | Source-only diagnostic entry; the explicit graph-blocked opt-in only reproduces the known all-zero logits from the third decode step |
| DINOv3 ViT-B | `examples/dinov3.py` | `dinov3` | Runnable vision entry |
| SigLIP | `examples/siglip.py` | `siglip` | Runnable component entry using the vision tower from a matching Pi0.5 bundle |
| Pi0.5 Libero FP16 | `examples/pi05.py` | `pi05` | Supported exact profile; 9/9 numeric/task and 27/27 execution-plan records pass, plus an independent fresh-process Graph lifecycle cell |
| Pi0.5 Libero W8A16 | `examples/pi05.py` | `pi05` | Source-only local-conversion entry; no public immutable derived checkpoint identity or hash is bound |
| Pi0.5 Libero W4A16 | `examples/pi05.py` | `pi05` | Source-only local-conversion entry; no public immutable derived checkpoint identity or hash is bound |
| Pi0.5 reduced-step / closed-loop | Public policy API | None | Source-only and validation pending; no public smoke claim and no inheritance from the exact FP16 or quantized paths |
| Pi0.5 base | None | None | Source-only alias; no independent task profile or smoke claim |
| Wall-OSS FP16/W8A16/W4A16 | `examples/wall_oss.py` | `wall_oss` | Public `wall-x` source-only integration |
| Hy-Embodied-0.5-VLA FP16/W16 | `examples/hy_embodied.py` | `hy_embodied` | Runnable normalized-action entry for the exact public profile |
| Hy-Embodied-0.5-VLA W8/W4 | `examples/hy_embodied.py` | `hy_embodied` | Source-only runtime-conversion entry; no public immutable derived checkpoint identity or hash is bound |
| RhinoVLA | `examples/rhinovla.py` | `rhinovla` | Integration entry; the model repository supplies its runtime factory |
| Gemma4-E4B text | `examples/gemma4.py` | `gemma4` | Source-only text entry; requires a compatible local checkpoint and dependency set |
| GR00T-N1.7-3B | `examples/gr00t.py` | `gr00t` | Source-only builder entry with caller-preprocessed tensor inputs |
| LingBot-VLA-V2 | `examples/lingbot2.py` | `lingbot2` | Public source-only integration with caller-supplied images |
| Galaxea G0.5 | `examples/g05.py` | `g05` | Configuration and integration-boundary check; execution fails closed until the model repository supplies the official CPU policy object |
| InternVLA-N1 + NavDP | `examples/internvla_navdp.py` | `internvla_navdp` | Controlled NavDP component with caller-pinned manifest and embeddings; not composite E2E |
| Qwen3 0.6B/1.7B/4B/8B W8A16, Qwen3 32B | Public API named in `model_support.md` | None | Source-only; no released runnable profile or smoke configuration |
| Unsupported profiles | None | None | Not runnable |

The table reports actual public entry points; source presence alone is not
listed as a test path.

## Checked-in profile templates

Every stem below has one `.toml` under
[`examples/configs`](../examples/configs). That file contains the direct model
request, runner environment, Torch/hardware profiling, and every accepted
execution stage. Configuration contents do not change the profile's support
status.

| Family | Configuration stems |
|---|---|
| Qwen3 | `qwen3_0_6b`, `qwen3_1_7b`, `qwen3_4b`, `qwen3_8b`, `qwen3_14b_w8a16` |
| Llama | `llama_3_2_1b` |
| Qwen3.5 text | `qwen3_5_0_8b`, `qwen3_5_2b`, `qwen3_5_4b`, `qwen3_5_9b` |
| Qwen3.5 Vision | `qwen3_5_vision_2b`, `qwen3_5_vision_4b` |
| Qwen3-VL | `qwen3_vl_2b`, `qwen3_vl_4b`, `qwen3_vl_8b`, `qwen3_vl_32b_w8a16` |
| DINOv3 / SigLIP | `dinov3_vit_b`, `siglip` |
| Pi0.5 Libero | `pi05_libero`, `pi05_libero_w8a16`, `pi05_libero_w4` |
| Wall-OSS | `wall_oss`, `wall_oss_w8a16`, `wall_oss_w4a16` |
| Hy-Embodied / RhinoVLA | `hy_embodied`, `rhinovla` |
| Gemma4 / GR00T | `gemma4`, `gr00t` |
| LingBot2 / G0.5 | `lingbot2`, `g05` |
| InternVLA-N1 + NavDP | `internvla_navdp` |

Quantized templates exercise local converter/runtime paths only. Their presence
does not bind a derived checkpoint identity or change a Source-only status.

## Direct examples

Every direct example accepts a TOML file and can validate it without loading a
checkpoint:

```bash
python examples/qwen3_vl.py --config examples/configs/qwen3_vl_2b.toml --check-config
python examples/qwen3_vl.py --config qwen3_vl.local.toml
```

Copy the closest file under `examples/configs/`, then set local checkpoint and
input paths. See [Model assets](model_assets.md) for checkpoint acquisition and
hashing.

The generic vision and VLM templates use `assets/example.ppm`, a tiny synthetic
image included only so the public entry points run without a private input.
Replace it with representative images before numerical or task validation.
VLA camera inputs remain caller supplied because their names and preprocessing
belong to the exact model profile.

The Pi0.5 `batch_file` is not bundled. Produce it with the checkpoint-compatible
LeRobot policy preprocessing pipeline, then save the tensor dictionary with
`torch.save`. At minimum it contains one batched image tensor for every key in
the checkpoint's `image_features` configuration, plus
`observation.language.tokens` and `observation.language.attention_mask`; the
preprocessing pipeline may retain other profile-owned fields. Camera names,
resolution, and token length are profile-owned; take them from the
checkpoint/configuration rather than copying another profile's shapes.

The GR00T entry similarly expects a caller-produced tensor mapping. Its
`input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, and `state`
must come from the checkpoint-compatible preprocessing pipeline. The NavDP
component entry requires a caller-pinned path-to-SHA256 JSON manifest plus a
tensor mapping containing `goal_embed [1,1,384]` and
`rgbd_embed [1,32,384]`. These files are inputs, not model assets distributed
by RhinoForge.

Qwen3.5 Vision rejects execution by default before loading model weights in the
public example and before creating its Vision handle or transforming Vision
weights in the shared adapter. For controlled evaluation only, explicitly set
the gate before starting a fresh process:

```bash
QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1 \
  python examples/run_model.py --config examples/configs/qwen3_5_vision_2b.toml
```

This does not waive the failed official real-image numerical gate or enable
video. The Qwen3-VL 32B template retains an explicit opt-in only to reproduce its
known hard Graph failure; it is not a passing hard-gate or runnable-support
claim. LingBot2 and InternVLA/NavDP templates contain explicit source-only or
controlled-evaluation acknowledgements. Removing a gate must fail
rather than fall back to an ordinary profile. G0.5
has no stable standalone public policy constructor: its example validates the
configuration and then stops with the exact integration step for the official
model repository. A successful `--check-config` for any of these entries
validates configuration structure only; it does not promote the support state.

## Complete TOML runner

Every runnable target has one TOML under `examples/configs/`. The same file
contains model/request fields, common runtime settings, Torch profile, hardware
profile, and every applicable `rpu_execution` stage. The complete field and
ownership map is in
[Runtime configuration](runtime_config.md#toml-parameter-catalog).

For example, start from `examples/configs/qwen3_0_6b.toml`:

```bash
python examples/run_model.py --list-targets
python examples/run_model.py \
  --config examples/configs/qwen3_0_6b.toml \
  --check-config
python examples/run_model.py \
  --config examples/configs/qwen3_0_6b.toml
```

The configuration combines runtime environment, per-handle execution planning,
model input, and Torch profiling:

```toml
[runner]
target = "causal_lm"

[runner.env]
RPU_LOG_LEVEL = 3
RPU_WARMUP = 1

[runner.torch_profile]
enabled = false
output = "profiles/torch.json"
record_shapes = false
profile_memory = false
with_stack = false
```

The remaining `[model]`, `[generation]`, or `[request]` tables are consumed by
the selected direct example. Only variables documented in
[Runtime configuration](runtime_config.md) are accepted under `[runner.env]`.
Boolean TOML values become portable `1`/`0` environment values. The runner
applies them before importing `torch` or `rpu_backend`; run one model/profile
per process. Credential variables, unrelated process-loader variables, and
`RPU_KERNEL_LIB_PATH` are rejected from TOML; set the approved Rhino Launch
library and combined operator asset in the deployment environment.
Variables under the Runtime configuration page's **Diagnostic-only inventory**
are also rejected; a developer must opt into one explicitly in the shell and
review any resulting local artifacts before sharing them.

Per-handle execution planning belongs in the top-level table:

```toml
[rpu_execution.prefill]
chunk_size = "auto"
padding_budget = 64
```

The comprehensive runner validates and forwards this table to the existing
model entry point. The accepted stages are:

| Target | Accepted `rpu_execution` stages |
|---|---|
| `causal_lm`, `qwen3_5_text`, `qwen3_5_vision` | `prefill` |
| `gemma4` | `prefill` (`chunk_size` only) |
| `qwen3_vl` | `prefill`, `vision` |
| `dinov3`, `siglip` | `vision` |
| `pi05`, `wall_oss`, `gr00t` | `prefill`, `vision`, `action` |
| `rhinovla` | Declared by the trusted runtime factory and checked by its capability handshake |
| `hy_embodied`, `lingbot2`, `g05`, `internvla_navdp` | None; a non-empty table is rejected |

Except for Gemma4's narrower row, `prefill` accepts `chunk_size`,
`padding_rows`, and `padding_budget`; `vision` and `action` accept
`chunk_size`. The public loader remains the authoritative profile-envelope
check and rejects unsupported values before RPU installation.

## Torch profile

Set `[runner.torch_profile].enabled = true` to export one Chrome trace for the
whole command. CPU and `PrivateUse1` activities are requested when the
installed PyTorch exposes both. The trace includes model initialization as well
as inference; filter for `rpu::` and named graph ranges when investigating the
device path.

`record_shapes`, `profile_memory`, and `with_stack` all default to `false`.
Enable them individually only when needed: shapes, allocation events, stack
frames, and source paths add application detail to the trace.

Open the JSON in Perfetto or another Chrome-trace viewer. Profiling adds
overhead, so use a profiler-disabled run for latency measurement.

## Debugging an incorrect or unstable run

Debug correctness before collecting performance traces. Keep one immutable
run record containing the checkpoint/configuration hashes, TOML, input tensors,
execution settings, and dependency versions. For a stochastic policy, sample
the initial noise or other random state once and clone that exact tensor into
the CPU and RPU paths; two successive RNG calls are different inputs.

Use the smallest public checks that distinguish the failure:

1. Compare the first call with a repeated call of the same signature. A
   repeat-only change points first to Graph lifecycle, persistent SPM, or DMA
   address ownership.
2. Retain the first output, run a different input of the same shape, and verify
   that the retained bytes and storage remain independent.
3. Inspect `GraphCache.snapshot()`, `cache_invariant_ok()`,
   `debug_bucket_counts()`, and `explain_miss(signature)` before changing the
   capture path. A clean log alone does not prove BUILD followed by REPLAY.
4. Isolate the first divergent public component or operator and compare its
   shapes, dtypes, layouts, and value against the same CPU input before
   debugging the full model.
5. Increase `torch.rpu.set_debug_level(...)` only as needed. Use
   `torch.rpu.spm_alloc_dump("label")` for allocator summaries and
   debug-tensor export only for a controlled local comparison.

Debug exports can contain weights, user-derived inputs, and activations. Keep
them outside the repository, clear them after the investigation, and share a
sanitized summary rather than the raw tensors. Once correctness and replay are
stable, rerun in a fresh process with only the profiler needed for the measured
question.

## What a release-quality model check still needs

For a support claim, bind all results to the same source commit, Rhino Launch
package, combined operator asset, checkpoint revision, input envelope, TOML,
and output hashes. Compare against a CPU or approved golden result, exercise
prefill and decode or the complete policy call, repeat after warmup, and verify
the applicable graph lifecycle. Record any unrun gate as pending rather than
inferring it from a smoke run. Use the independent hard semantic, same-dtype
parity, FP32-anchor, and representative task gates in
[Model validation policy](validation_policy.md); a single worst-row cosine is
diagnostic evidence, not a repository-wide product verdict.
