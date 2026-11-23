# Knowledge workflows

## Query

1. Start at [INDEX.md](INDEX.md).
2. Read the smallest relevant concept or synthesis page.
3. Follow its public source links before changing code or making a support
   claim.
4. Use `rg` only after the indexed sources do not answer the question.

See [RETRIEVAL.md](RETRIEVAL.md) for the Markdown-first retrieval contract and
the boundary for optional semantic indexing.

For build-version mismatches, compare the installed Launch package with
`release/runtime-dependencies.json`. RhinoForge's development version does not
determine the Launch version. Preserve exact dependency and capability checks;
see the [release policy](../docs/release_policy.md).

For a clean installation, use the [unified installation and development guide](../docs/getting_started.md):
install Miniconda3, create one `RHINOFORGE_ENV`, and keep it for source, editable
and wheel operations. The [dependency environment bundle](../docs/environment_bundle.md)
replaces third-party dependency installation only: clone and build RhinoForge
separately in the same environment. It excludes RF, Launch, operators, all board
SDK libraries and models. Use `pack_dependency_env.py` and the standalone
`install_dependency_env.py`, then check source/environment compatibility with
`dependency_environment.py` before building RF.
Validate relocation and source/dependency compatibility before publication.
The recommended validated board SDK is EA_R1SDK_2026_08F-2156007a, distinct from
Launch 1.2.0; use [runtime extraction and platform checks](../docs/runtime_assets.md)
and the first-run guide's platform device lock. Clear inherited Python
and native-library overrides, bind compiler and asset identities, and distinguish
the new application environment from the board OS/SDK/driver baseline.
`--check-config` is device-free; it does not prove runtime availability.
For CausalLM permission failures, retain the failing node and OS error and verify
failure before weight loading or transformation. Preserve CPU-only loading and
restart the process after repairing an import-time native-load failure. Normal
Graph BUILD/REPLAY is informational; do not classify all native stderr as an
error or downgrade genuine errors. See [API behavior](../docs/api_reference.md#rpumodelforcausallmfrom_pretrained)
and [model diagnostics](../docs/model_testing.md).

For model assessment or implementation, use the
[`rhinoforge-port` skill](../.agents/skills/rhinoforge-port/SKILL.md) with
[model porting](../docs/model_porting.md).

For TTS codec migration, check the installed native capabilities and the
[speech guide](../docs/qwen3_audio.md#experimental-rpu-vocoder-convolutions).
Compare fixed codec tokens, not just equal generation lengths. Keep padded
residual rows through device addition and crop through the owned CPU output;
validate repeated outputs when changing a device layout or FP16 boundary.

## Add or update knowledge

1. Confirm the statement in public source or documentation.
2. Update the existing page that owns the fact; create a page only for a new,
   reusable concept.
3. Follow [SCHEMA.md](SCHEMA.md), add direct source links, and update
   [INDEX.md](INDEX.md).
4. Check relative links and scan the changed files for private infrastructure,
   credentials, restricted implementation detail, opaque asset internals, and
   diagnostic payloads.

## Save a reusable answer

1. First update an existing concept or synthesis page when it owns the result.
2. Add a short page under [queries/](queries/README.md) only when the original
   question and decision context remain useful.
3. Keep direct public source links and mark unresolved points; do not store raw
   chat transcripts or diagnostic artifacts.

## Add an external source

Prefer a stable public URL. When an offline snapshot materially improves
provenance or availability, follow [raw/README.md](raw/README.md), confirm its
license or redistribution basis, and cite it from a maintained page.

## Resolve conflicts

For runnable examples, consult the [configuration index](../examples/configs/README.md)
and validate the selected file with `python -m examples.run --config <TOML> --check-config`.
Keep one template per model/version/component/precision/workload. Put TOML files
in their model-family directories and runner implementations in `examples/runners/`.
Update the template, documentation and configuration checks together. Use the
current paths from the index in commands and file-copy instructions.
Keep persistent model construction outside inference mode; apply the selected mode to inference only. Warm a new Graph with BUILD and REPLAY. With `warmup >= 2`, omit `warmup_decode_steps` or match `input.decode_steps`; fixed-length runs use `stop_on_eos=false`.
The [LingBot v2 runner](../examples/configs/lingbot/v2/README.md) permits `warmup=0` after its A/B/A and READY preparation; retain an independent first output and compare subsequent replays against it.
The [exact-Z2 entry](../examples/configs/lingbot/legacy/exact_z2/README.md) uses its checkpoint processor and its own input template; keep it separate from the v2 replay runner.
The [HALO example](../examples/configs/halo/README.md) resolves `halo_action_expert`
directly to `HALO/action_expert.safetensors` under the model cache. Check the
paired single-step fixture before weight loading; do not append checkpoint directories.
Count retained, temporary-capture and validation queues against the same cold arena pool. Snapshot policy is per-instance and frozen; unsupported SDK/asset combinations use ordinary replay. See [runtime configuration](../docs/runtime_config.md).
For cache-maintenance changes, trace the actual CPU reads and writes through
their producing copy operations. Device-only handoffs retain storage and DMA
ownership without republishing data; host-backed device copies still need both
boundaries. Check in-place request updates and mask changes before reusing a
cached upload. See [component handoffs](concepts/component-handoff.md).
For cache-capacity changes, distinguish allocator capacity from execution
admission across supported adapters. Bind requested model capacity, rotary
tables, physical prefill padding and fixed-address keepalives during cold
installation; require cache capacity to fit the installed model. Public
CausalLM/ConditionalGeneration loaders accept `max_seq_len`, as do Pi0.5 policy
constructors and the TTS talker; GR00T exposes `max_prefix_len`. Check default,
explicit and deferred device-installation paths. Keep actual cache bounds,
rotary indices, coordinate/register widths, SPM feasibility and semantic input
shapes checked before state mutation. Test allocation failure without silently
shortening the request. Defaults are not hard allocation limits, and expanded
allocation does not expand hardware-validation claims.
See [KV cache](concepts/kv-cache.md).
For Pi0.5 fixed-input comparisons, use the public `input.noise` field for both
Graph preparation and inference; retain default-input failures separately.
See the [policy API](../docs/api_reference.md#policy-apis) and
[model family map](synthesis/model-family-map.md#pi05).
For RhinoVLA v3 RTC, bind `rtc` at construction and validate the paired
`action_prefix`/`delay` request before model loading. Preserve per-token time
conditioning and the state token's mean time embedding, refresh prefix contents
on replay, and include delay in Graph identity. Verify the requested delay range
independently of non-RTC evidence. Reserve the complete Graph pool during cold
construction, including transient preparation while all delay variants remain
retained. Keep the Python/native RTC build paired; VLA dependencies in an older
bundle do not supply the new runtime. Use the [RTC input guide](../examples/configs/rhinovla/v3/README.md#rtc)
for the exact FP16/W8A16 envelope and current verification status. Keep RTC
orthogonal to the existing expert/full-expert/full-pipeline precision recipe;
preserve its fusion/resident/HIGH/vector/partial-RoPE choices. Prepare delay
tables with the selected condition-projection weights and scales, and verify
installed storage for RTC profiles just as for their non-RTC counterparts.
When adding quantized examples, trace the checkpoint loader and installed
projection storage, not just the `dtype` label. Document the component scope
and retain its admission controls. Bind the recipe, input and lifetime requirements before extending a profile. For caller factories, bind precision inputs,
reject conflicting factory settings and verify the installed model before
inference. See [quantization](../docs/quantization.md).

Use public API documentation for user-visible behavior and source for actual
implementation. Treat model-support status as profile-specific. Use the entry
points and runtime boundaries linked from [Examples](../docs/model_support.md)
and follow [model validation policy](../docs/validation_policy.md). Update stale
knowledge in the same change; do not preserve conflicting summaries.

## Qwen3.5 MoE mixed-E4M3 weights

Download the official unquantized `Qwen/Qwen3.5-35B-A3B` and run
`python/rpu_backend/quant/convert_qwen3_5_moe_fp8.py` on CPU using the
[configuration guide's commands](../examples/configs/qwen3_5/README.md#cmd-moe-mixed).
Write to `$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-rpu-fp8-mixed` or select the output
explicitly with `--model`. No calibration is required; official block-FP8,
GPTQ and RTN cannot replace the unquantized source. When changing this workflow,
update the weight index and bilingual quantization, asset and API guides,
and check the loader's `quant_scope.py` contract. Preserve the recorded identity
of historical assets and measurements.

## Speech changes

Start with the [Qwen3 speech guide](../docs/qwen3_audio.md) and retain these contracts:

- Validate frozen checkpoints and prepared TTS source hashes before loading.
  Construct TTS on CPU and load safetensors strictly, preserving nonpersistent
  FP32 RoPE buffers. Apply compatibility edits only to the prepared source copy.
- ASR uses explicit SigLIP audio mode: fourteen heads padded to sixteen,
  896-wide residuals, ERF GELU, original FP16 clamp and head scale. The current
  audio Graph includes both output projections and their GELU; its profile hash
  and physical plan include that tail. Keep model-owned staging stable and refresh
  the final output's mutable DMA address per call. Eager-only wrappers cannot be
  inserted directly into capture.
- ASR clears old text Graphs before audio execution on a prompt-length change.
  Preserve CPU FP32 convolution of FP16-rounded weights and FP16 output boundaries.
  Prefill input composition stays on device; single-token embedding lookup stays on CPU.
  Enable the fused output head only for causal-LM callers and retire the decoder
  before releasing that head.
- TTS Talker, Predictor and optional codec share one owner policy and six arenas.
  Enable `causal_decoder_set_tts_codec_mode` only on the installed codec handle;
  its default-off state preserves other decoders' RoPE parameters and layout hash.
  Keep original CPU FP32 codec weights; fold LayerScale into new frontend copies
  before FP16 conversion and preserve FP32 rotary frequencies. Use the device
  frontend for 1–255 codec frames; above 64, plan MASK_2D single-chunk execution
  with the original 72-position causal window and a versioned owned CPU mask.
  Declare RoPE/KV/envelope 256 before first planning. Evict the old Graph and
  clear its signature before replacing the mask, including failed length switches.
  Missing setter selects CPU; report the route and propagate errors.
  New transpose postprocessing uses kernel-major weights and bounded single-core
  SPM last-axis slices, not DDR transpose. Keep 768/384-channel residual chains
  in device rows. With `tts_codec_residual_unit_padded_rows`, keep 192/96-channel
  chains in zero-padded 256/128-channel rows too: pad each kernel position's input
  columns before transforming the first convolution weight, and crop only at stage
  exit. Preserve the original per-unit weights for older-extension CPU boundaries.
  With `tts_codec_decoder_stage_rows`, pass owned contiguous FP16 device rows
  between stages 1–3; stage 3 crops its padded channels and returns CPU NCT before
  stage 4. Keep each helper's eager completion and temporary-SPM lifetime intact;
  do not place the whole chain inside a Graph with stack-owned mutable DMA slots.
  The optional `tts_codec_conv_transpose1d_device_post` boundary reuses shared-DDR
  conversion helpers and the existing projection/overlap math. Validate the full
  geometry before upload, and return independently owned CPU FP32 NCT storage.
  Codec SnakeBeta retains its five rounded stages in per-core temporary SPM;
  split rows across cores, bound scratch to 4 MiB and tiles to 512 rows/core,
  and upload alpha/scale once per invocation. Group at most eight original tiles
  per one-shot Graph, as for overlap/bias postprocessing and im2col fill/DMA/expansion.
  Upload overlap bias in the first batch. ERF GELU likewise batches at most eight
  25,600-element tiles without changing its per-tile math or scratch size.
  Preserve im2col padding, dilation and its 4 MiB capacity plan. Finish/abort and retire
  the Graph before releasing its SPM scope; never persist these temporary
  addresses across requests. Compare fixed inputs, tail tiles and retained outputs.
  ConvNeXt LayerNorm batches at most sixteen original 16-row tiles, reusing its
  scratch only in stream order; refresh affine parameters on each invocation.
  Codec-only FP16/ACC32 Linear uses the shared launcher in a one-shot Graph,
  retaining input, weight, optional bias and fresh output through completion.
  Keep single-row scatter eager and retain the existing segment resource budgets.
- Reset logical KV position per sequence. Talker keeps EOS and repetition penalties;
  Predictor uses a fixed 15-step sampling loop and resets KV per frame. Both use
  top-k 50, temperature 0.9 and top-p 1.0. Require EOS before waveform decoding;
  each segment allows up to 256 steps. Compare deterministic outputs after
  resetting the same seed when changing sampling.
- With `tts_predictor_generate`, run the fixed Predictor loop natively only after
  both exact signatures are BUILT and replayable. Keep embedding, FP32 top-k and
  default-generator multinomial on CPU, preserving token order and RNG state.
  Release Python lookup references before execution; native shared ownership ends
  on return or failure. Abort failed replays, retain partial logical counters and
  propagate errors without resampling. This reduces Python dispatch, not the
  fifteen serial device round trips per frame; older extensions use the Python loop.
- The external-asset device-sampling experiment keeps centering and finite/range
  validation on-device. Sanitize invalid logits before top-k/softmax/gather, retain
  every step's validity flags, and reject the group on readback before returning
  tokens. Verify NaN/Inf and both envelope endpoints on the target operator assets;
  do not assume device comparison semantics match CPU. This removes per-step
  logits readback, not the remaining decoder/sampler submissions. Predictor and
  Talker load their kernels from `tts_sampling_unified.ref` and their separately
  sized table sections from the versioned `tts_sampling_tables.bin`; point both
  sampling asset environment variables at that same two-file directory.
- The optional TTS Predictor group path records decoder, sampler and embedding
  into one 15-step Graph under its typed cold TP8/six-arena owner policy. Raise
  only `qwen3_tts_predictor_group`'s entry ceiling to 16,384; retain SDK headroom
  and KD/instruction limits. Other owners retain 8-step + 7-step grouping.
  Include group length in the signature and retire the old two step Graphs first; keep Programs, per-invocation mutable DMA slots
  and managed initialization/noise buffers alive until group retirement. Disable
  the single-forward prefill shortcut at installation, and do not use it to skip
  a group stream. Check the **delta** of `hw_batch_submit_total` around each end;
  it is cumulative across BUILD/REPLAY. Each group must submit once; per-request
  totals exclude Talker, Codec and boundary copies. Compare speech/RNG and CPU
  core-seconds as well as wall time; reduced submissions alone are not acceptance.
- Predictor checked decode replay retains the ordinary input/output route and
  sampling. Bind its exact five-layer FP16/32-row/1D-RoPE grammar, versioned
  immutable weights and tables, KV owners, Graph lifetime and physical layout
  before skipping host emission. Refresh position and attention fields on every
  replay, including resets to zero; use ordinary emission when the proof cannot
  be established, and fail explicitly on inconsistencies after skipping starts.
- Predictor binds fifteen FP16 `1024→2048` heads once and appends its ACC16 head
  to the ACC32 body Graph. Prefix output uses logical row 1; decode uses head
  index `position - 1`. Bind head-bank identity in the physical profile and prove
  all owners/versions before patching the head's Linear DDR registers on replay.
  Keep owned logits, hidden outputs and the original sampling sequence; consume
  fused logits once and only for the matching hidden object and head index.
  Older native extensions retain the single-row head's one-shot Graph. Keep operands alive through
  synchronous completion, retain DMA/flush checks and abort on failure; never
  reuse a Graph with fresh fixed input addresses. CPU codec temporary reuse must
  preserve residual inputs, retained outputs and the frozen SnakeBeta formula.
- Codec Conv1d shared-DDR staging uses fresh device allocations: fill the FP16
  im2col destination through a retained CPU view, force publication, run the
  existing synchronous ACC32 Linear, then read through the coherent CPU view.
  Crop padded channels before the single owned NCT FP32 conversion/layout copy.
  Keep FP32 bias arithmetic, reject capture, and propagate errors. Compare the
  same codec inputs against the staged path; never expose writable CPU aliases
  or replace device computation with a CPU matrix multiply.
- Keep every Predictor token on device. Its frozen third MLP layer (index 2)
  overflows FP16 at `SiLU(gate) * up`, before the down projection; subsequent
  RMSNorm receives Inf and produces zero. The ordinary device RMSNorm handles
  the finite ~53k residual correctly in isolation, so do not infer a norm defect
  from a CPU FP16-square experiment. Divide up weights by 8 and multiply down
  weights by 8 in FP32 before packing, without mutating CPU reference weights.
  Check the paired transform's semantics, finite intermediate products, and
  position-zero device decode/replay. Keep sentence-stop-only tails attached
  within the segment bounds so punctuation cannot exhaust EOS.
  Explicitly bind Talker/Predictor FP32 Linear accumulation before planning their
  descriptors; changing it after descriptor resolution leaves a stale plan.
- Accept up to 1024 characters. Keep requests within 128 characters and 32 tokens
  on their single-segment path. Split longer text at punctuation with a 12-token
  target and 128-character bound; retain tiny sentence endings up to 16 tokens.
  Use sentence stops and the request-initial CPU RNG state for each segment.
  Later segments restore the caller RNG state left by the first; return owned
  concatenated audio only after every segment succeeds.
- Keep `stream_synthesize()` on the same segment generation path. Hold the model
  request lock across the generator lifetime, but leave inference mode before
  yielding each owned waveform. Early-stop callers close the generator; failure
  and cancellation release the lock and retain only completed-segment statistics.
  Keep `synthesize()` as the no-partial-output consumer of that generator.
- Bind output-head reuse to the exact current hidden object and consume it once.
  Predictor borrows the final device row for its immediate head; requests for
  hidden states retain owned CPU outputs. Never retain a borrowed row across
  another replay. Cache only the latest decoded reference speaker embedding.
- Talker reuses the 28-layer/1024 prefill envelope within its 512-position cache.
  Exclude physical padding from logical KV position; replace only the old prefill
  Graph when prompt length changes. Predictor batches only its two initial
  conditioning tokens at logical position zero using a 32-row/32-position
  envelope, then keeps single-token checked decode. Exclude padded rows from
  the returned head input and logical KV position. Retain separate Predictor
  prefill/decode Graphs and all 15 dependent sampling/head steps; codec keeps
  one current-frame Graph. Retire Graphs before handles and weights.
- Speech TOMLs under `examples/configs/qwen3_asr` and `qwen3_tts` use the shared
  CLI and the dedicated audio runner. Keep configuration checks free of Torch/device
  imports and validate paths/audio before loading weights. Record resident latency,
  process CPU time and actual-duration RTF separately from load and profiling.
- TTS defaults to `codec_backend="rpu"` in both the API and TOML runner;
  explicit `codec_backend="cpu"` restores FP32 Conv1d. The RPU path offloads all
  31 decoding Conv1d layers through synchronous FP16/ACC32 Linear, including
  two depthwise layers expanded to channel-diagonal weights at installation.
  Leave the shared quantizer's two encoder-only input projections untouched. Validate
  all geometries before conversion, preserve CPU weights, restore forwards on close
  and report actual calls. CPU padding/im2col, remaining bias/residual arithmetic,
  transposed convolutions, ConvNeXt Linear/normalization and 17 SnakeBeta layers
  remain explicit with native pairs enabled; do not claim
  full device execution or bitwise equality with the FP32 anchor.
- Native codec residual pairs keep two ACC32 Linear projections and their
  intermediate FP16 bias/SnakeBeta on device. Bind frozen parameters and padded
  second weights once; preserve FP32 final bias/residual on CPU without an extra
  residual upload. Reject unsupported geometry before transformation. One pair
  contributes two convolution calls; compare the changed precision boundary
  using fixed codec tokens and speech content, not a bitwise FP32 assertion.
  Do not use eager legacy LocalSPM LayerNorm inside a model that already owns
  the arenas: ConvNeXt migration needs framework-owned SPM and lifecycle planning.
- CPU codec matrix multiplication and adjacent-block overlap-add preserve output
  lengths. Keep thread settings caller-controlled and check retained outputs, input changes and close/reload
  when changing component handoffs.

## Version and release changes

Update [VERSION](../VERSION) and the matching Launch dependency lock together.
Run `python scripts/check_release_version.py` and the affected package checks.
Cut `release/X.Y` before advancing `main` to the next `.dev0` version. Preserve
published tags and assets; record each source commit and payload checksum.
See the [release policy](../docs/release_policy.md).

## Tensor diagnostics and cached masks

Keep diagnostic graph modes separate from normal retained graphs and bind cold
choices before installing a native owner. Test mode drift, frozen-cache misses
and retirement. Tensor getters share storage; callers keep a replay-independent
value with `.cpu().clone()`. Cached attention masks must compare actual input
contents, including inference tensors, and own versioned CPU storage.
