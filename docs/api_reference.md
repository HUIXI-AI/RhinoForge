# API reference

[简体中文](api_reference.zh.md) | English

The distribution is named `rhinoforge`; its Python package remains
`rpu_backend`. Importing the package registers PyTorch's `PrivateUse1` device as
`rpu` and creates the `torch.rpu` namespace.

Choose the checkpoint, precision and input profile from [examples](model_support.md).

## Top-level package

`rpu_backend.__all__` contains exactly four names:

| Name | Contract |
|---|---|
| `__version__` | Installed RhinoForge distribution version |
| `RPUCache` | FP16 KV cache for fused causal attention |
| `RPUModelForCausalLM` | Loader for registered decoder-only CausalLM profiles |
| `reset_graph_cache()` | Clear the process graph cache; a no-op without the native backend |

The API compatibility level is available as `rpu_backend.api.API_VERSION` and
currently reports `5.0.0`; it is independent of the distribution version.

The top-level package also lazily exposes `set_debug_level(0..5)`,
`get_debug_level()` and `TRACE`, mirrored on `torch.rpu`. These names are outside
`__all__`; other diagnostics use `torch.rpu`. Levels are SILENT, ERROR, WARN,
INFO, DEBUG and TRACE, with INFO as the default.

### Tensor diagnostics

`torch.rpu.set_debug_export(True)` enables supported intermediate tensor
captures. Set the mode before constructing and capturing the model. Use
`list_debug_tensors()` and `get_debug_tensor(name)` to read them;
`get_debug_export()` returns the current mode. The getter returns shared
storage: use `.cpu().clone()` to keep values across later replays.
`clear_debug_tensors()` releases registry entries; disabling export also clears
them. NextDiT binds the mode for its lifetime. Qwen3.5 keeps separate diagnostic
prefill graphs and rejects a missing diagnostic graph after Graph freeze.
Pi0.5 can also save named input and intermediate tensors as `.pt` files when
`RPU_PI05_PROBE_DIR` is set to a local output directory.

## Causal language models

```python
import torch

from rpu_backend import RPUCache, RPUModelForCausalLM

model = RPUModelForCausalLM.from_pretrained(
    checkpoint,
    dtype=torch.float16,
    device="rpu",              # omit to load on CPU, then call model.to("rpu")
    rpu_execution={
        "prefill": {
            "chunk_size": "auto",
            "padding_rows": "auto",
            "padding_budget": 64,
        }
    },
)
cache = RPUCache.from_model(
    model,
    input_ids=input_ids,
    max_new_tokens=32,
)
```

### `RPUModelForCausalLM.from_pretrained`

```python
RPUModelForCausalLM.from_pretrained(
    hf_repo_or_path,
    *,
    dtype=torch.float16,
    device=None,
    rpu_execution=None,
    quantization=None,  # CausalLM: explicit dense Qwen3 W4/G32 installation recipe
    **hf_kwargs,
)
```

The loader reads the Hugging Face configuration before the full weight load,
selects an adapter by `config.architectures[0]`, and rejects unsupported
profiles early. It currently admits the built-in Qwen3 and Llama CausalLM
architectures. Only `torch.float16` is accepted by this entry point; a
quantized checkpoint carries its own checked metadata.

`rpu_execution` is a read-only per-handle mapping. The CausalLM entry accepts a
`prefill` stage with:

- `chunk_size`: `"auto"` or a positive multiple of 16;
- `padding_rows`: `"auto"` or a non-negative exact row count; and
- `padding_budget`: a non-negative integer, mutually exclusive with an exact
  `padding_rows` value.

For Qwen3, `prefill.linear_acc32` selects ACC16 (`false`, default) or ACC32
(`true`). The cold root `model.num_cores` budget is `4`, `6` or `8`;
the selected checkpoint recipe determines which budgets are accepted.
Cold fields require a fresh model; supported planning fields use the execution
update API below. Qwen3-VL, Qwen3.5 and Pi0.5 expose their
component precision and topology controls through the same mapping; see the
[target-specific configuration catalog](runtime_config.md#configuration-and-execution).

`trust_remote_code` defaults to `False` and is forwarded to Hugging Face
loaders. `Pi05Policy.from_pretrained` warns when it is enabled. Adapter model
and checkpoint checks still apply.

### `RPUCache`

Prefer `RPUCache.from_model` over the low-level constructor:

```python
RPUCache.from_model(
    model,
    *,
    input_ids=None,
    max_new_tokens=...,
    max_seq_len=None,
    batch_size=1,
    device="rpu",
)
```

Pass exactly one sizing form:

- `input_ids` plus `max_new_tokens`; or
- an explicit positive `max_seq_len`.

The cache exposes `reset()`, `reset_to_position(pos)`, `get_seq_length()`,
`get_max_length()`, `get_cache(layer_idx)`, and `to_dynamic_cache(device="cpu")`.
Batch sizes above one are limited to the explicitly admitted Qwen3 path and
require equal real-token lengths across rows.

`rpu_backend.api.Qwen3_5Cache.from_config(text_config, max_seq_len, ...)` adds
the recurrent and convolution state used by Qwen3.5 text. It may reset fully to
position zero but cannot reconstruct a non-zero recurrent state by rewinding.
For a loaded or reduced-core model, prefer
`Qwen3_5Cache.from_model(model, max_seq_len=...)` so KV and convolution storage
match its installed topology. A cache created from configuration alone must
use the same core budget as its model.

### Greedy token selection

```python
from rpu_backend.api import greedy_token_ids

out = model(**inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
next_ids = greedy_token_ids(out.logits)
```

`greedy_token_ids` accepts `[batch, vocab]` or `[batch, sequence, vocab]`, using
the last sequence row. It returns independent CPU `int64 [batch, 1]` storage
and leaves logits unchanged. CPU FP16/BF16/FP32/FP64 retain their precision;
device logits must be FP16. Equal maxima select the first index, and a row
with NaNs selects the first NaN, matching `torch.argmax`. Empty batches are
accepted; empty sequence/vocabulary dimensions and other ranks or dtypes fail.
The caller handles cache positions, EOS and sampling policy; this helper does
not change Hugging Face `generate()`.

Qwen3 RoPE capacity comes from the installed tables or the model's declared
`max_position_embeddings`. A request must fit both the cache and those tables;
increasing cache capacity alone does not extend the model context.

## Execution configuration and updates

Pass `rpu_execution` as a Python mapping, or load the same mapping from TOML/JSON. Component overrides take priority over top-level stage defaults, which take priority over profile defaults. The model keeps an immutable copy. An integer `chunk_size` requests that exact size; `"auto"` selects within the admitted range. An integer `padding_rows` fixes padding and cannot accompany `padding_budget`.

Owners with a live execution session expose shared configuration updates and statistics:

```python
from rpu_backend.api import reconfigure_rpu_execution, rpu_execution_stats

before = rpu_execution_stats(model)
reconfigure_rpu_execution(model, {
    "prefill": {"chunk_size": "auto", "padding_budget": 64},
})
after = rpu_execution_stats(model)
```

Updates replace the hot fields of each supplied stage, including stages nested under `components`; omitted stages stay unchanged. `{"prefill": {}}` clears its planning overrides while preserving bound cold fields such as `linear_acc32`. An unchanged configuration is a no-op. A committed change advances `generation`, retires affected graphs, and rebuilds them on the next execution. Call between requests. Missing sessions, active forwards, fixed/unsupported controls and closed or failed sessions reject updates. Core count, precision and other cold installation settings require a fresh model. Failed retirement or rollback requires a fresh process. See [configuration lifecycle](../knowledge/concepts/runtime-profiles.md#safe-profile-changes).

## Image-text models

`rpu_backend.api.RPUModelForConditionalGeneration` is the Hugging Face-style
loader for Qwen3-VL, Qwen3.5 Dense and the exact Qwen3.5 MoE profiles.
A Qwen3-VL example:

```python
from rpu_backend.api import RPUModelForConditionalGeneration

model = RPUModelForConditionalGeneration.from_pretrained(
    checkpoint,
    dtype=torch.float16,
    device="rpu",
    rpu_execution={
        "prefill": {"chunk_size": "auto"},
        "vision": {"chunk_size": "auto"},
    },
)
```

It has the same CPU-first loading and fail-fast profile behavior as the
CausalLM loader. The `vision` stage accepts `chunk_size`; the `prefill` stage
accepts the fields described above. Video and profiles outside
[Examples](model_support.md) are not implied by this class.

Qwen3.5 Dense uses this loader for text-only and image-text requests; its
adapter is selected from checkpoint metadata. Gemma4 has a separate adapter
entry. See [model entries](model_support.md) and [compute topology](planner_core_topology.md).

### Qwen3.5 27B

Official 27B uses uniform W8 Text/VL at 4/6/8 cores or W4 group32 at 8 cores. Root6 assigns MLP to 6 cores and attention/GDN/head/Vision to 4. It requires the existing quantization opt-in and, for image inputs, the Vision opt-in. It uses caching allocation, four cold Graph arenas, finalized uploads from CPU weights, physical KV24, bounded one-shot prefill and retained decode. FP16 runtime is not provided. [Configuration and weights](../examples/configs/qwen3_5/README.md).

### Qwen3.5 MoE

The 35B-A3B RTN/GPTQ paths use the conditional loader, MoE adapter and hybrid cache. Select the checkpoint matching the [configuration](../examples/configs/qwen3_5/README.md); RTN and GPTQ metadata are not interchangeable.

```python
import torch
from rpu_backend.api import RPUModelForConditionalGeneration, Qwen3_5MoeCache
from rpu_backend.adapters.qwen3_5_moe import Qwen3_5MoeAdapter

model = RPUModelForConditionalGeneration.from_pretrained(
    checkpoint, dtype=torch.float16, local_files_only=True,
    rpu_execution={"model": {"num_cores": 8}, "prefill": {"chunk_size": 64}},
)
model = Qwen3_5MoeAdapter(model).to_rpu(max_seq_len=1024)
cache = Qwen3_5MoeCache.from_model(model, max_seq_len=1024, dtype=torch.float16)
with torch.no_grad():
    output = model(input_ids=input_ids, past_key_values=cache,
                   use_cache=True, return_dict=True, logits_to_keep=0)
cache.reset()
```

`input_ids` is CPU `int64` with shape `[1, T]`; these profiles use eight cores. Install once, reuse the cache during decoding, and call `reset()` before a new request. Prefill uses a bounded single graph and decoding retains its graph. Choose the prefill chunk size and cache capacity for the request length.

For image input, pass the processor's `input_ids`, `attention_mask`, `pixel_values` and `image_grid_thw` together on the first call after reset. Keep subsequent calls text-only while reusing that cache. Returned logits are CPU FP32.

Prepare the separate mixed-E4M3 profile from the official unquantized
`Qwen/Qwen3.5-35B-A3B` source using
[`convert_qwen3_5_moe_fp8.py`](../python/rpu_backend/quant/convert_qwen3_5_moe_fp8.py).
The [download, CPU conversion and run commands](../examples/configs/qwen3_5/README.md#cmd-moe-mixed)
write `$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-rpu-fp8-mixed`; pass that output to
`from_pretrained` or the example's `--model`. Conversion needs no calibration
and produces per-output E4M3 weights. Official block-FP8 checkpoints are not
accepted inputs or substitutes for this format. Select the matching mixed-E4M3
Text/VL and core-count template; the RTN/GPTQ limits above remain separate.

## Policy APIs

The following classes are exported from `rpu_backend.api`:

| Class | Primary construction and inference methods |
|---|---|
| `Pi05Policy` | `from_pretrained(...)` or `from_lerobot_policy(...)`; `to("rpu")`; `prepare_graphs(...)`; `predict_action_chunk(...)`; `select_action(...)` |
| `RhinoVLAPolicy` | `from_runtime(...)`, `from_factory(...)`, or `from_pretrained(..., runtime_factory=...)`; `prepare_graphs(...)`; `predict(...)`; `predict_action_chunk(...)` |
| `WallOssPolicy` | `from_checkpoint(...)` or `from_pretrained(...)`; `to("rpu")`; `prepare_graphs(...)`; `infer(...)`; `predict_action_chunk(...)` |
| `Lingbot2Policy` | `from_checkpoint(...)`; `to("rpu")`; `prepare_graphs(...)`; `infer(...)`; `predict_action_chunk(...)`; `close()` |
| `HyEmbodiedPolicy` | `from_checkpoint(...)`; `to("rpu")`; `infer(...)`; `predict_action_chunk(...)`; `close()` |

Pi0.5 [templates](../examples/configs/pi05/README.md) select precision,
camera count and language length. The runner binds cold settings before loading
weights and passes `rpu_execution` to `Pi05Policy.from_pretrained(...)`.
Checkpoint metadata must match the selected precision. Use
`prepare_graphs(..., precompute_adarms=...)` to select AdaRMS preparation explicitly.

`Pi05Policy.prepare_graphs(batch, noise=...)` and
`Pi05Policy.predict_action_chunk(batch, noise=...)` accept an optional initial
noise tensor. It must be finite CPU `float32` with shape
`[batch_size, chunk_size, max_action_dim]` from the loaded policy configuration.
Use the same tensor for graph preparation and inference when comparing fixed
inputs. Omitting it preserves the policy's normal seeded noise generation.
Pi0.5 example TOML files expose this as `input.noise`, a path to a tensor saved
with `torch.save`; the optimized examples require shape `[1, 50, 32]`.

Pi0.5 `prepare_graphs()` reports Graph readiness and replay diagnostics.
It checks shape, dtype, finite values and Graph invariants.
`replay_parity.numerical_status` is `DIAGNOSTIC`; `task_quality` is `NOT_EVALUATED`.

For the release-bound Pi0.5 component profiles, pass `profile="pi0.5-base"`
or `profile="pi0.5-libero-v044"` together with the first preprocessed
`admission_batch`. The loader verifies the exact public payload and request
envelope before loading weights; subsequent Graph and inference requests are
checked against the same profile.

Keep a loaded `HyEmbodiedPolicy` or `WallOssPolicy` instance in the application and call `infer(...)` for successive requests. Serialize access to each instance; refresh request inputs between calls. Call `HyEmbodiedPolicy.close()` when finished.

`WallOssActionOutput`, `Lingbot2ActionOutput`, and
`HyEmbodiedActionOutput` are the corresponding structured return types. A
physical-unit action field applies the configured normalization.

`WallOssPolicy.from_checkpoint(...)` binds the checkpoint, camera roster,
normalization and precision; `.to("rpu")` installs the runtime. Supply images
in the same order as `camera_names` when calling `infer(...)`. The policy
validates actual request geometry and state against the installed model.
`prepare_graphs(...)` optionally binds an explicit finite prefix range for
representative inputs; reuse a range only for inputs admitted by that policy.

`RhinoVLAPolicy` deliberately delegates checkpoint composition and preprocessing
to an explicit model-repository runtime factory. The runtime must expose the
execution capabilities and effective `rpu_execution` configuration it
consumed; the facade rejects a silent mismatch.

A string `runtime_factory="module:callable"` imports and executes Python code
from that module. Treat it as a trusted extension point: use only an installed,
reviewed model integration and do not copy the value from an untrusted TOML or
request. Passing an already imported callable has the same trust requirement.

`HyEmbodiedPolicy.from_checkpoint` does not read `norm_stats.pkl` by default.
Physical-action decoding requires `norm_stats_path`, the explicit
`trust_norm_stats_pickle=True` opt-in, and `norm_stats_sha256` for the exact
file bytes. The hash is checked before those same bytes are deserialized.
The fixed three-camera facade accepts `prefix_len` only in
`{192, 208, 224, 240}`. Request images, noise, state embeddings/state, and end
effector pose must be finite; invalid values are rejected before policy
execution.

Use each profile's documented constructor, input schema and precision.

### LingBot v2 runtime Graph

Use `rpu_backend.adapters.lingbot_vla_v2.runtime.build_lingbot_vla_v2(..., graph_runtime_policy=...)` and `get_action` for the current FP16/W8A16 entry. The [input guide](../examples/configs/lingbot/v2/README.md) specifies two/three cameras, training YAML, language storage and normalized `[1,50,55]` outputs. The older `Lingbot2Policy.from_checkpoint` facade accepts `rpu_execution` and retains the separate exact-Z2 profile.

## Error hierarchy

All user-facing errors are available from `rpu_backend.api`:

```text
RPUBackendError
├── RPUConfigError
├── PlannerRejectError
├── UnsupportedModelError
├── RPUUnsupportedDtypeError
├── RPUSingleHandleError
├── SPMExhaustionError
└── WeightShapeMismatchError
```

Catch `RPUBackendError` for backend-level failures, or a subclass when the
caller has a specific recovery action. `PlannerRejectError` exposes `.code`
(`EXACT_MISMATCH`, `CAPABILITY`, `NO_FEASIBLE` or `SEARCH_LIMIT`) and `.stage`
for a rejected execution plan; it also remains catchable as `RuntimeError`.
Standard `TypeError` and `ValueError` remain in use for malformed Python arguments.

## Graph API

`GraphRuntimePolicy` accepts cold `batch_snapshot_mode="auto"` or `"off"`. `GraphCache` and the Pi0.5, Wall-OSS and LingBot v2 builders accept the policy before weight installation. AUTO uses supported SDK/asset capabilities and otherwise falls back to ordinary replay; OFF preserves Graph execution. See [snapshot policy](runtime_config.md#graph-snapshot-policy).

The public graph namespace is `rpu_backend.graph`:

```python
from rpu_backend.graph import GraphCache, GraphSignature

cache = GraphCache(max_entries=4)
sig = GraphSignature(
    op_id="my_model_forward",
    shapes=(seq_len, hidden_size),
    dyn_dims=(num_layers,),
    dtypes=(torch.float16,),
)

with cache.capture(sig):
    output = torch.ops.rpu.my_model_forward(...)
```

`GraphCache` provides `capture`, `begin_warmup`, `freeze`, `is_frozen`,
`lookup`, `evict`, `clear`, `size`, `max_entries`, `snapshot`, and
`cache_invariant_ok`. A frozen cache is lookup-only and rejects an online
BUILD. Most applications should let the model adapter own its graph cache;
manual graph construction is primarily a porting interface.

`Graph.runtime_policy` and `GraphCache.runtime_policy` are read-only views of
the policy bound at construction. Extracted child graphs inherit that policy.
After a failed capture, the scope releases its borrowed graph reference so
`evict()` or `clear()` can run while the exception traceback is retained.
References explicitly retained by callers still prevent graph retirement.

User-level diagnostics remain public. On a `GraphCache` instance,
`debug_bucket_counts()`, `debug_branch_counts()`, `dump_signature_tree()`, and
`explain_miss(signature)` report cache organization, signature state, and miss
reasons. `Graph.debug_stats()` returns an aggregate mapping of execution
counters for the most recent graph submission. These methods are useful for
lifecycle diagnosis, but their output is not numerical or support evidence.
`Graph.dump_replay_plan()` returns a pointer-free linear node/segment summary;
`Graph.dump_tree()` presents the same execution structure grouped by segment.
`Graph.dump_resources()` lists resource owners, byte counts, lifetimes, and pool slots.
`Graph.dump_kernel_args_jsonl()` returns kernel capacity metadata from the Release runtime; raw state is unavailable.
`Graph.debug_nodes()` and `Graph.debug_segments()` return lists of dictionaries;
`Graph.debug_segments_jsonl()` returns segment records as JSONL. Records contain
node kinds and dependencies, logical tensor metadata, transfer lengths and
segment structure. All these views omit raw addresses and launch argument words.

`Graph` is the lower-level capture object and `get_default_graph_cache()`
returns a thread-local cache. New adapters should normally use a dedicated
per-model `GraphCache` so ownership and teardown are explicit.

## Adapter registry and plugins

The registry lives in `rpu_backend.runtime.registry`:

```python
from rpu_backend.runtime.registry import (
    get_adapter,
    list_adapters,
    register_adapter,
)
```

Built-in architecture bindings are declared in
`rpu_backend.adapters._manifest.BUILTIN_ADAPTERS`. The current manifest binds:

- `Qwen3ForCausalLM`;
- `LlamaForCausalLM`;
- `DINOv3ViTModel`;
- `Gemma4ForConditionalGeneration`;
- `Qwen3_5ForConditionalGeneration`;
- `Qwen3_5MoeForConditionalGeneration`; and
- `Qwen3VLForConditionalGeneration`.

An external distribution may declare an entry point in the
`rpu_backend.plugins` group. The loaded callable receives a small namespace
whose `register_adapter(architecture, adapter_class)` method adds bindings:

```toml
[project.entry-points."rpu_backend.plugins"]
my_adapter = "my_package.rpu_plugin:register"
```

```python
def register(api):
    api.register_adapter("MyModelForCausalLM", MyModelAdapter)
```

Plugins are loaded lazily after an unknown-architecture lookup, or explicitly
with `discover_plugins()`. They cannot override a built-in architecture or a
previously registered plugin binding. Loading an entry point executes code from
the installed plugin distribution; install and enable only plugins you trust.

The registry is process-wide and has no public unregister operation. Once a
built-in or plugin architecture is bound, keep that binding for the process
lifetime; start a fresh process after changing the installed plugin set.

## Model-path registry

`rpu_backend.model_registry` provides local path resolution:

```python
from rpu_backend.model_registry import MODELS, cache_root, model_path

path = model_path("qwen3-0.6b")
```

`cache_root()` uses `RPU_MODEL_CACHE` when set and otherwise resolves to
`~/.cache/rhinoforge/models`. `model_path(name)` appends the checked-in relative
path and raises `KeyError` for an unknown alias. An alias is only a path mapping;
it is not a support or asset-availability claim.

## Weight-layout API for adapter authors

The canonical helpers are in `rpu_backend.runtime.weights`:

- `get_linear_partition(in_features, out_features, dwidth=2, num_cores=8)`;
- `swizzle_linear(weight, partition, num_cores=8)`; and
- `swizzle_model_inplace(model, ..., skip_names=...)`.

Swizzling changes parameter storage in place and must run exactly once. A
specialized linear owned by a fused subsystem belongs in `skip_names` and must
be transformed by that subsystem's one authoritative conversion path.

The existing dense FP16 Qwen3-8B loader validates all original weights before
installation, reserves its large vocabulary head and embedding first, then
converts and transfers one Linear at a time. It releases CPU weight storage
between transfers to reduce simultaneous CPU/device allocation. The public loading
API is unchanged. A failed partial installation still requires a fresh model.

## `torch.rpu` device controls

The commonly useful device-level functions include:

- `is_available()`, `device_count()`, and `current_device()`;
- `set_caching_allocator(bool)`, `empty_cache()`, and `memory_stats()`;
- `set_ddr_flush(bool)` and `set_ddr_flush_force(bool)`;
- `set_debug_level(0..5)` and `get_debug_level()`;
- `set_cross_layer_batch_prefill(bool)` and
  `set_cross_layer_batch_size(int)`; and
- `shutdown()`.

Developer diagnostics are also available through `torch.rpu`:

- `set_debug`, `get_debug`, `set_profile`, `get_profile`, and
  `reset_profile_accumulators`;
- `set_spm_debug`, `get_spm_debug`, and `spm_alloc_dump`.

These diagnostics expose logs and aggregate counters, not model inputs,
intermediate activations, raw addresses, or Graph register/resource payloads.

`set_caching_allocator(bool)` is a cold, process-wide tensor-storage choice.
The first explicit call, or the first non-empty device tensor allocation when no
choice was made, freezes the policy for the process. Repeating the same choice
is idempotent; requesting the opposite choice raises an error and requires a
fresh process. Call it only before device tensor materialization. `empty_cache()`
releases unused cached blocks and cannot release live tensor or Graph-owned
storage. `memory_stats()` reports the selected mode in
`caching_allocator_enabled` and whether it is frozen in
`caching_allocator_policy_frozen`.

CPU/device boundary flushing is enabled by default and should not be disabled in
normal inference. Chunk size is a per-handle `rpu_execution` setting, not a
process-global `torch.rpu` setter. Internal `torch.ops.rpu.*` schemas are host
wrapper interfaces used by adapters; unless a function is documented here or
in a model example, treat it as an implementation detail.

## Bounded speech APIs

```python
from rpu_backend.api import Qwen3ASRModel, Qwen3TTSModel, WhisperASRModel
```

| API | Input and result |
|---|---|
| `Qwen3ASRModel.from_pretrained(checkpoint, dtype=torch.float16)` | Local pinned Qwen3-ASR-0.6B checkpoint |
| `model.transcribe(audio, sample_rate=16000, max_new_tokens=64)` | One finite CPU floating-point waveform, mono 16 kHz, 0.5–8 seconds; token limit 1–64. Returns `ASRTranscription(text, raw_text, token_ids)` |
| `WhisperASRModel.from_pretrained(checkpoint, dtype=torch.float16)` | Local pinned `sherpa-onnx-whisper-base` checkpoint containing encoder, decoder and token table |
| `model.transcribe(audio, sample_rate=16000, max_new_tokens=448)` | One finite non-silent CPU waveform up to 30 seconds; resamples and detects language automatically. Returns `WhisperTranscription(text, language, token_ids, finish_reason)`; the 448-position context includes four prompt tokens, leaving at most 444 content tokens |
| `Qwen3TTSModel.from_pretrained(checkpoint, source=prepared_package, dtype=torch.float16, codec_backend="rpu")` | Local pinned Qwen3-TTS-12Hz-0.6B-Base checkpoint and prepared `qwen_tts/` source directory |
| `model.synthesize(text, reference_audio=local_file, language="Chinese", max_new_tokens=256)` | Text of 1–1024 characters; `Chinese` or `English`; finite, nonzero mono 24 kHz reference audio, 0.5–20 seconds; token limit 2–256 per segment. Returns an independent float32 mono waveform and sample rate 24000 |

ASR uses greedy generation without context prompts, timestamps or streaming.
TTS uses non-streaming speaker-embedding-only voice cloning with the pinned
checkpoint sampling recipe: top-k 50, temperature 0.9 and top-p 1.0. Set
`torch.manual_seed` after loading and before a request to repeat sampling on the
same execution profile. Talker prompt plus generation must fit 512 positions
per segment; reaching the token cap without EOS raises before waveform decoding.
Empty, non-finite and exactly zero waveforms are rejected.

Inputs within 128 characters and 32 tokenizer tokens use one segment. Longer
text splits at punctuation with a 128-character bound and a 12-token target;
a short sentence ending may join its preceding clause up to 16 tokens. Adding
sentence stops can change token counts, pauses and intonation. Each segment
starts with the request's initial CPU RNG state; later segments restore the
caller RNG state left by the first. Each waveform is copied independently;
concatenation waits for all segments to succeed. A failed request returns no
partial audio. Downloads,
preparation and runnable API examples are in the [speech guide](qwen3_audio.md).

All three APIs support `with`, idempotent `close()` and `execution_stats()`. Serialize
requests, statistics and close, with one live device model/policy per process.
Loaders validate assets before transformation and select the caching allocator
before device allocation. TTS also declares six owner-specific arenas. Restart
if the process has an incompatible frozen allocator or arena plan.

ASR requires `siglip_set_audio_mode`, `siglip_set_audio_projection` and
`assemble_inputs_embeds` in the rebuilt native extension. TTS codec uses the
explicit `causal_decoder_set_tts_codec_mode`; without it, the frontend stays on
CPU. With the setter, 1–63 codec frames use the device; 64–255 frames use the
original CPU FP32 frontend and its 72-frame sliding attention. Other execution
errors propagate.

`execution_stats()` exposes call and cache state:

- ASR keeps one audio Graph for the current audio-token length and at most the
  current prompt length's prefill/decode pair. `audio_graph_size`,
  `audio_graph_replays` and `audio_graph_invariant_ok` describe the audio cache;
  `graph_size`, `graph_replays` and `graph_invariant_ok` describe the text cache.
  Replay totals reset when cached signatures are replaced. `audio_linear_calls`
  counts logical projections.
- Whisper reports the retained encoder, prompt-prefill and decode Graph sizes,
  replay counts, invariants, the one-time `build_ms`, and the current request's
  `replay_ms`. Cross-K/V preparation runs once per audio input.
- TTS reports `generation.steps`, `generation.eos` and `generation.segment_steps`;
  `codec_frontend.device` and `fallback_reason` describe the last segment. It also
  reports output-head calls and each component's calls, tokens and Graph state.
  `components.predictor.fused_head_calls` counts logits consumed from the fused
  body/head Graph; `head_calls` still counts all logical head invocations.
  Talker replaces its prefill Graph on a prompt-length change while retaining
  decode; Predictor retains its two-token conditioning prefill and one
  decode Graph; codec replaces its Graph on a frame-count change.

Each sequence resets logical KV position while retaining private storage. TTS
caches only the latest reference speaker embedding by decoded audio content;
every request still reads and validates the file. Output storage is independent
of replay. Predictor keeps its final device hidden row only until its immediate
output head consumes it; requests for hidden states retain independent CPU
outputs. Retire Graphs before native handles and weights. Set CPU thread count
with `torch.set_num_threads` before loading; the API leaves it unchanged.

`codec_backend="rpu"` defaults to experimental offload of all 31 decoding Conv1d
layers. With matching native support, twelve pairs also keep their intermediate
SnakeBeta and first bias on RPU in FP16. Select `codec_backend="cpu"` to restore
CPU FP32 convolutions and activations. `execution_stats().codec_convolutions`
reports backend, installed layers, cumulative convolution calls, `transport`
(`shared_ddr` or older `staged`), `residual_pairs` and completed `paired_calls`.
Each pair contributes two convolution calls. See the [speech guide](qwen3_audio.md#experimental-rpu-vocoder-convolutions)
for remaining CPU work and the changed numerical boundary.
