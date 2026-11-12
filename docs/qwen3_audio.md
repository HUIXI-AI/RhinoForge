# Qwen3 ASR / TTS guide

[简体中文](qwen3_audio.zh.md) | English

`Qwen3ASRModel` transcribes local audio. `Qwen3TTSModel` generates speech from
text and reference audio. Both retain dedicated Python APIs and now have TOML entry points under
[ASR examples](../examples/configs/qwen3_asr/README.md) and
[TTS examples](../examples/configs/qwen3_tts/README.md), dispatched by
`python -m examples.run --config PATH`. They do not use the CausalLM/VLM registry.

## Models and inputs

| Asset | Pinned source |
|---|---|
| ASR checkpoint | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0) |
| ASR reference source | [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff); inference uses Transformers components directly and does not require this checkout |
| TTS checkpoint | [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/tree/5d83992436eae1d760afd27aff78a71d676296fc), including `speech_tokenizer/` |
| TTS source | [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/tree/022e286b98fbec7e1e916cb940cdf532cd9f488e), prepared as described below |

| Model | Inputs | Generation and output |
|---|---|---|
| ASR | Batch one; finite mono floating-point samples at 16 kHz, 0.5–8 seconds | Greedy, `max_new_tokens=1..64`; text, raw text with language metadata, and token IDs; no context prompt, timestamps or streaming |
| TTS Base | Batch one; `Chinese` / `English`; 1–1024 text characters; finite, nonzero mono 24 kHz reference audio, 0.5–20 seconds | Whole-waveform or segment-level streaming sampling, `max_new_tokens=2..256` per segment (default 256); independent float32 mono waveform chunks at 24000 Hz |

TTS uses only the reference speaker embedding. ICL transcripts, CustomVoice and
VoiceDesign are unavailable. Talker prompt plus the generation limit must fit
512 positions per segment. Sampling follows the pinned checkpoint: top-k 50,
temperature 0.9 and top-p 1.0. Missing EOS raises before waveform decoding; empty,
non-finite or exactly zero output is rejected.

Inputs within 128 characters and 32 tokenizer tokens use one segment. Longer
text splits at punctuation with a 128-character bound and a 12-token target.
A sentence ending of at most eight characters can join its preceding clause up
to 16 tokens. Adding sentence stops can change token counts, pauses and
intonation. Every segment starts from the request's initial CPU RNG state; later
segments restore the caller state left by the first. All segments must succeed
before their independently stored waveforms are concatenated.
Sentence-stop-only tails join the preceding segment within those bounds.
`stream_synthesize()` yields each completed segment instead. It is not acoustic
frame streaming: a short request that stays in one segment still yields one
chunk after that segment finishes.

Loaders check the pinned checkpoint, configuration and tokenizer before weight
transformation. ASR identities are in
[`qwen3_asr.py`](../python/rpu_backend/api/qwen3_asr.py); TTS identities are in
[`_manifest.py`](../python/rpu_backend/adapters/qwen3_tts/_manifest.py).
Keep the original tokenizer files without overrides. Both APIs use eight device
cores and accept only `dtype=torch.float16`.

## 1. Install

Create a Python 3.12 Conda environment using the [unified installation guide](getting_started.md)
and configure [matching runtime assets](runtime_assets.md). From the repository
root, install the `audio` extra:

```bash
python -m pip install '.[audio]' --no-build-isolation \
  -Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  -Cbuild.tool-args=-j2
python -m pip check
```

[pyproject.toml](../pyproject.toml) pins the dependencies, including PyTorch 2.10.0
and Transformers 5.3.0. Keep this Transformers version and prepare the TTS source
without installing the upstream TTS package. The `audio` extra includes
TorchAudio, ONNX Runtime and python-sox for the upstream import chain. The current
12Hz path does not invoke the SoX executable; its absence may cause an upstream warning.

## 2. Download weights and prepare TTS source

```bash
export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
mkdir -p "$RPU_MODEL_CACHE"
hf download Qwen/Qwen3-ASR-0.6B \
  --revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-ASR-0.6B"
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --revision 5d83992436eae1d760afd27aff78a71d676296fc \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-TTS-12Hz-0.6B-Base"

export QWEN3_TTS_SOURCE=/path/to/Qwen3-TTS
export QWEN3_TTS_PREPARED=/path/to/qwen3-tts-prepared
git clone https://github.com/QwenLM/Qwen3-TTS.git "$QWEN3_TTS_SOURCE"
git -C "$QWEN3_TTS_SOURCE" checkout 022e286b98fbec7e1e916cb940cdf532cd9f488e
python -m rpu_backend.adapters.qwen3_tts \
  --source "$QWEN3_TTS_SOURCE" --output "$QWEN3_TTS_PREPARED"
cp "$QWEN3_TTS_SOURCE/LICENSE" "$QWEN3_TTS_PREPARED/LICENSE"
```

Replace `/path/to/...` with local directories. The prepared output directory must
not exist. Preparation copies 15 hash-pinned source files and applies compatibility
patches without executing the original source or modifying its checkout. Loading
checks the prepared manifest and file hashes. The API `source` argument points
to the generated `qwen_tts/` subdirectory. Keep the upstream license with the copy.

## 3. Run

Use an idle device with the platform's device lock. Run ASR and TTS in separate
processes.

The ASR API reads mono floating-point samples at 16 kHz. This example reads a
local WAV and prints the transcription as JSON:

```python
import json
import os
from dataclasses import asdict
from pathlib import Path

import soundfile as sf
import torch
from rpu_backend.api import Qwen3ASRModel

torch.set_num_threads(4)
checkpoint = Path(os.environ["RPU_MODEL_CACHE"]) / "Qwen3-ASR-0.6B"
audio, sample_rate = sf.read("/path/to/mono-16k.wav", dtype="float32")
with Qwen3ASRModel.from_pretrained(checkpoint, dtype=torch.float16) as model:
    result = model.transcribe(audio, sample_rate=sample_rate, max_new_tokens=64)
print(json.dumps(asdict(result), ensure_ascii=False))
```

The TTS API accepts a local reference file. This example writes a new FLOAT WAV;
the output directory must exist:

```python
import os
from pathlib import Path

import soundfile as sf
import torch
from rpu_backend.api import Qwen3TTSModel

torch.set_num_threads(16)
checkpoint = Path(os.environ["RPU_MODEL_CACHE"]) / "Qwen3-TTS-12Hz-0.6B-Base"
source = Path(os.environ["QWEN3_TTS_PREPARED"]) / "qwen_tts"
with Qwen3TTSModel.from_pretrained(checkpoint, source=source, dtype=torch.float16) as model:
    torch.manual_seed(0)
    waveform, sample_rate = model.synthesize(
        "Hello, welcome.", reference_audio="/path/to/mono-24k.wav",
        language="English", max_new_tokens=256,
    )
with open("/path/to/output/hello.wav", "xb") as output:
    sf.write(output, waveform, sample_rate, format="WAV", subtype="FLOAT")
```

To consume completed text segments as soon as they are ready, use the Python
generator and close it when stopping early:

```python
stream = model.stream_synthesize(
    "A longer sentence to synthesize.",
    reference_audio="/path/to/mono-24k.wav",
    language="English",
)
try:
    for waveform, sample_rate in stream:
        play(waveform, sample_rate)
finally:
    stream.close()
```

The generator holds the model request lock from its first iteration until it
finishes, fails or is closed. A later segment failure cannot retract chunks the
caller already consumed. `synthesize()` continues to return no partial audio.

Set the process thread count for your host before loading; the APIs leave it
unchanged. Set the seed after loading and before each request to repeat sampling
on the same execution profile. Reuse the model context for multiple requests.
ASR also accepts a one-dimensional CPU floating-point Tensor. See
[speech APIs](api_reference.md#bounded-speech-apis) for signatures and return types.

## Execution and resources

| Model | CPU | Rhino RPU |
|---|---|---|
| ASR | Audio reading, mel features/chunking, convolutions and their activations, single-token embedding, finite checks, argmax, text decoding | conv_out, positions and valid rows, audio encoder and output projections, prefill text/audio composition, text decoder and vocabulary head |
| TTS | Embedding, reference processing, speaker encoder, generation control, codebook, ConvNeXt normalization/GELU, transposed-convolution overlap/layout, remaining 17 SnakeBeta layers, waveform output | All 31 decoding Conv1d layers, four ConvNeXt Linear layers, six transposed-convolution projections and 12 intermediate SnakeBeta layers with the matching native extension; Talker, Predictor and output heads; codec input/output projections and Transformer frontend when supported by native interfaces |

ASR convolutions use channels-last storage and FP16-rounded weights in CPU FP32
computation, then round outputs to FP16. The device audio Graph uses FP32 Linear
accumulation. TTS CPU components stay FP32; device components use FP16 with FP32
Linear accumulation, except Talker/Predictor output heads, which retain FP16
accumulation. Codec LayerScale is folded into projections in FP32 before
conversion to FP16, while RoPE retains FP32 frequencies. The device codec frontend
accepts 1–63 frames. Sequences of 64–255 frames use the original CPU FP32 frontend
and its 72-frame sliding attention.

- ASR requires a rebuilt native extension exposing `siglip_set_audio_mode`,
  `siglip_set_audio_projection` and `assemble_inputs_embeds`; loading fails if
  these interfaces are absent.
- TTS explicitly enables the device codec path with
  `causal_decoder_set_tts_codec_mode`. Without that setter, the frontend stays on
  CPU. `execution_stats()` reports the last segment's device and fallback reason,
  total generation steps, EOS and per-segment steps. Other execution errors
  propagate to the caller.
- Keep one live model/policy per process. Serialize requests, `execution_stats()`
  and model `close()`. Both models support `with`, idempotent `close()` and outputs with
  storage independent of later replay.
- Loaders select the caching allocator before device allocation. TTS requests
  six owner-specific arenas: five retained Graphs and one serial immediate call.
  Start a new process if its frozen allocator or arena plan is incompatible.
- ASR retains one audio Graph and at most two text Graphs for current lengths.
  TTS retains Talker prefill/decode, Predictor prefill/decode and the current codec frame
  count. Components evict old signatures when their length changes and replay
  matching signatures.
- Predictor processes its two conditioning tokens together at logical KV position
  zero, using the existing bounded prefill planner and a 32-position/32-row
  envelope. Padding never advances logical KV or supplies the output-head row.
  Each frame needs 15 body executions instead of 16; all 15 dependent output
  heads and the sampling recipe remain unchanged. Later tokens use checked decode replay.
- With a rebuilt native extension, each Predictor body and its `1024→2048`
  output head share one Graph submission. Installation binds fifteen FP16 heads;
  the body retains ACC32 and the head retains ACC16. The two-token prefix selects
  logical row 1, and decode selects its head from the logical KV position.
  Logits have independent storage and are consumed once by the matching head;
  callers requesting hidden states still receive owned CPU output. No persistent
  Graph is added. `components.predictor.fused_head_calls` counts consumed fused
  results; `head_calls` continues to count logical head calls.
- Older native extensions retain the separate head path: its single-row TP8
  FP16/ACC16 projection batches upload, compute and write-back in a one-shot Graph.
  Both paths retain DMA ownership, boundary flushes and completion checks.
- CPU codec SnakeBeta precomputes the frozen parameters' exponentials and inverse
  scale, then updates only a newly allocated temporary in place. Convolutions add
  bias into their own output. Residual inputs and previously returned waveforms
  remain independent, and CPU arithmetic stays FP32.
- All Predictor tokens, including the first conditioning token, execute on RPU.
  In the frozen checkpoint's third layer (index 2), `SiLU(gate) * up` can exceed
  FP16 range before the down projection, although the final residual fits FP16.
  Installation divides that layer's up-projection weights by 8 and multiplies
  its down-projection weights by 8 in FP32 before FP16 packing. The paired
  power-of-two scaling preserves the MLP function without clipping, per-token
  work or a CPU decoder path; original checkpoint/CPU weights stay unchanged.
  Talker and Predictor explicitly bind FP32 Linear accumulation before resolving
  their Graph descriptors; their activation and weight storage remains FP16.
- Predictor uses native checked decode replay only for its five-layer FP16 path,
  including FP32 Linear accumulation, with a 32-position cache and 1D RoPE.
  Fused heads additionally bind all fifteen weight owners and the ACC16 head
  kernel, refreshing its DDR register pair for every decode position.
  The recorded Linear kernels must match each site's accumulation mode.
  After recording the full
  operator sequence, replay checks weight versions, KV owners, Graph identity and
  memory layout, then updates position and attention parameters while skipping
  repeated host emission. Input uploads, device computation, output heads and
  sampling remain unchanged. An unproven binding uses ordinary execution. Rebuild
  the matching native extension; generic Graph skip controls are not a substitute.
- TTS caches only the latest reference audio's speaker embedding by decoded
  content; every request still reads and validates the file. After a failed
  transformation, close the model and reload clean weights. Retire Graphs before
  releasing handles and weights.

## Experimental RPU vocoder convolutions

`Qwen3TTSModel.from_pretrained` and the example runner default to
`codec_backend="rpu"`, including when the argument/configuration field is omitted.
Both `examples/configs/qwen3_tts/0_6b/base.toml` and `base_rpu_codec.toml` select it.
Set `codec_backend="cpu"` explicitly to restore CPU FP32 Conv1d.
This option moves all 31 Conv1d layers used in decoding to explicit eight-core RPU
Linear with FP16 inputs/weights and FP32 accumulation: twelve kernel-7 residual
convolutions, fourteen kernel-1 projections, three input/output convolutions and
two depthwise convolutions. Depthwise weights expand into a channel-diagonal
matrix at installation, preserving group semantics. The shared quantizer's two
encoder-only input projections are not transformed. Weights are converted once
from independent copies; original CPU FP32 weights remain intact. Standalone
convolution outputs are padded to 128 alignment and sliced back before applying
the original FP32 bias.

CPU still performs padding, im2col, remaining bias/residual arithmetic, codebook,
17 SnakeBeta layers, ConvNeXt normalization/GELU and upsampling layout
operations. Older native extensions also keep all six transposed convolutions on CPU.
With native `tts_codec_conv1d`, CPU im2col writes rounded FP16 values directly into
fresh device-owned DDR through the existing storage-retaining CPU view. After
synchronous ACC32 Linear, the coherent shared-DDR output is cropped to valid
channels and copied once into owned NCT FP32 output, combining conversion and
layout. Standalone convolutions preserve FP32 bias arithmetic. No CPU alias escapes the op;
input publication and output coherency flushes remain mandatory, and Graph capture
is rejected. Older native extensions retain the staged FP16 im2col path.
With native `tts_codec_conv_pair`, each of the twelve residual units keeps its
kernel-7 projection, first bias, intermediate SnakeBeta and kernel-1 projection
on device. Channel padding remains through both projections; installation pads
both axes of the second weight and binds frozen FP16 bias/alpha/scale copies.
The first SnakeBeta, causal padding, final FP32 bias and residual addition remain
on CPU. There is one shared-DDR input preparation and one owned CPU output per
pair, with no intermediate CPU activation or extra residual upload. The two
Linear calls retain ACC32; intermediate arithmetic is FP16 and changes the
waveform relative to the prior CPU FP32 activation path. Compare identical codec
tokens and representative speech content, not bitwise equality with that path.
The pair uses existing eager kernels, not a single fused device program or Graph;
per-kernel submits and boundary flushes remain. Older native extensions keep all
SnakeBeta layers on CPU. `codec_backend="cpu"` keeps both convolutions and
activations on CPU.

With native `tts_codec_conv_transpose1d`, the six frozen transposed-convolution
projections use FP16 inputs and weights with ACC32 Linear on RPU. CPU retains
overlap addition, FP32 bias and final layout, so no device Graph is introduced.
The two ConvNeXt blocks also run their four pointwise Linear layers on RPU with
FP16 inputs/weights and ACC32. LayerNorm and exact GELU remain CPU FP32 to avoid
the v1.2.0 core-0 SPM bound; each Linear returns an independent CPU FP32 tensor.

This remains partial vocoder offload. Each Linear
uses the existing synchronous immediate path and the shared sixth arena; no new
retained Graph is introduced. Unsupported geometry rejects installation; execution
errors propagate instead of silently reverting to CPU. `execution_stats()` adds
`codec_convolutions` with `backend`, installed `layers` (31 in RPU mode), cumulative
RPU `calls`, `transport` (`shared_ddr` or the older `staged` route),
`residual_pairs` (12 with the matching native extension, otherwise 0), and
`paired_calls` (completed pair executions; each also adds two to `calls`), plus
`transposed_layers` (6 with matching native support), `transposed_calls`,
`convnext_linears` (4) and `convnext_linear_calls`.
The frontend's existing 63-frame device bound and CPU sliding-attention fallback
remain unchanged.

FP16 rounding changes the waveform. Compare identical codec tokens against the
CPU FP32 anchor and inspect speech content; runtime checks alone do not certify
audio quality. The RPU path remains experimental even though it is now the default; it does
not inherit the previous CPU residual-convolution profile's quality evidence.
