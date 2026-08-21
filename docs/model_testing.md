# Model execution and profiling

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
| Qwen3 FP16, Qwen3 14B W8A16, Llama-3.2-1B | `examples/causal_lm.py` | `causal_lm` | Runnable with a matching local checkpoint/profile |
| Qwen3.5 text | `examples/qwen3_5_text.py` | `qwen3_5_text` | Runnable text entry |
| Qwen3-VL 2B/4B | `examples/qwen3_vl.py` | `qwen3_vl` | Runnable image-and-text entry |
| DINOv3 ViT-B | `examples/dinov3.py` | `dinov3` | Runnable vision entry |
| Pi0.5 Libero FP16/W8A16 | `examples/pi05.py` | `pi05` | Runnable with a matching checkpoint and caller-provided tensor batch |
| Pi0.5 W4 / reduced-step / closed-loop | `examples/pi05.py` | `pi05` | Controlled profiles only; no general public smoke configuration |
| Pi0.5 base | None | None | Source-only alias; no independent task profile or smoke claim |
| Wall-OSS-0.5 | `examples/wall_oss.py` | `wall_oss` | Runnable with the named checkpoint sidecars and inputs |
| Hy-Embodied-0.5-VLA | `examples/hy_embodied.py` | `hy_embodied` | Runnable normalized-action entry |
| RhinoVLA | `examples/rhinovla.py` | `rhinovla` | Integration entry; the model repository supplies its runtime factory |
| Qwen3 0.6B/1.7B/4B/8B W8A16, Qwen3 32B | Public API named in `model_support.md` | None | Source-only; no released runnable profile or smoke configuration |
| Qwen3.5 Vision, Qwen3-VL 32B, Wall-OSS RTC/W4, LingBot-VLA-V2, G0.5, InternVLA-N1/NavDP | Public API named in `model_support.md` | None | Controlled or experimental; no general public smoke profile |
| SigLIP | Component API named in `model_support.md` | None | Component-only; no standalone end-to-end runner |
| GR00T-N1.7-3B, Gemma4-E4B | Public adapter API named in `model_support.md` | None | Source-only; public asset/end-to-end setup is incomplete |
| Unsupported profiles | None | None | Not runnable |

The table reports actual public entry points; source presence alone is not
listed as a test path.

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

The Pi0.5 `batch_file` is not bundled. Produce it with the checkpoint-compatible
LeRobot policy preprocessing pipeline, then save the tensor dictionary with
`torch.save`. At minimum it contains one batched image tensor for every key in
the checkpoint's `image_features` configuration, plus
`observation.language.tokens` and `observation.language.attention_mask`; the
preprocessing pipeline may retain other profile-owned fields. Camera names,
resolution, and token length are profile-owned; take them from the
checkpoint/configuration rather than copying another profile's shapes.

## Complete TOML runner

Start from `examples/configs/qwen3_0_6b_full.toml`:

```bash
python examples/run_model.py --list-targets
python examples/run_model.py \
  --config examples/configs/qwen3_0_6b_full.toml \
  --check-config
python examples/run_model.py \
  --config examples/configs/qwen3_0_6b_full.toml
```

The configuration combines runtime environment, per-handle execution planning,
model input, and two profiling layers:

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

[runner.hw_profile]
enabled = false
output_dir = "profiles/hw"
max_dumps = 32
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
| `causal_lm`, `qwen3_5_text` | `prefill` |
| `qwen3_vl` | `prefill`, `vision` |
| `dinov3` | `vision` |
| `pi05`, `wall_oss` | `prefill`, `vision`, `action` |
| `rhinovla` | Declared by the trusted runtime factory and checked by its capability handshake |
| `hy_embodied` | None; a non-empty table is rejected |

`prefill` accepts `chunk_size`, `padding_rows`, and `padding_budget`; `vision`
and `action` accept `chunk_size`. The public loader remains the authoritative
profile-envelope check and rejects unsupported values before RPU installation.

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

## RPU hardware profile

Set `[runner.hw_profile].enabled = true` to enter
`torch.rpu.hw_perf_trace(...)` around the same command. The backend writes up
to `max_dumps` hardware trace files to `output_dir`. Enabling tracing does not
retroactively rebuild an adapter-owned graph, so use a fresh process and turn
it on before model construction, `prepare_graphs`, or the first forward/Graph
BUILD. The runner establishes that ordering and disables tracing on normal or
exceptional exit.

Raw hardware JSON includes DMA source/destination device addresses and exact
execution structure. Torch traces include operator names and timing, plus any
explicitly enabled shapes, memory events, or stack frames. Treat both as
sensitive application artifacts; inspect them before sharing and do not commit
generated `profiles/` directories.

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
