# Model entries and execution scope

[简体中文](model_support.zh.md) | English

Choose a TOML from [model examples](../examples/README.md), prepare its checkpoint and inputs, then run `python -m examples.run --config PATH --check-config` before execution. This command validates configuration only.

Each profile specifies checkpoint, precision, core count, input shape and length. Loaders and runtime admission enforce those limits; controlled entries also require the template's opt-ins. See the [configuration index](../examples/configs/CONFIGURATIONS.md).

Supply model weights and request data separately. Follow [model assets](model_assets.md) and [runtime assets](runtime_assets.md) for dependencies, and [running models](model_testing.md), [API](api_reference.md) and [quantization](quantization.md) for usage.

## Model and input map

These entries link to the executable configuration and input contract. A family
name alone does not specify its checkpoint, precision or supported shapes.

| Model | Entry and guide | Inputs |
|---|---|---|
| Qwen3 / Llama | `RPUModelForCausalLM`; [Qwen3](../examples/configs/qwen3/README.md), [Llama](../examples/configs/llama/README.md) | Tokens and matching KV cache |
| Qwen3-VL | `RPUModelForConditionalGeneration`; [guide](../examples/configs/qwen3_vl/README.md) | Text, images and processor-produced positions/grid metadata |
| Qwen3.5 dense / MoE | `RPUModelForConditionalGeneration`; [guide](../examples/configs/qwen3_5/README.md) | Text or image/text; matching dense or MoE cache and checkpoint recipe |
| Whisper base | `WhisperASRModel`; [guide](../examples/configs/whisper_asr/README.md) | Mono audio up to 30 minutes; automatic language, greedy, batch 1, one-second streaming updates |
| Gemma4 | `Gemma4Adapter`; [guide](../examples/configs/gemma4/README.md) | E4B Instruct text generation; CPU frontend and device decoder |
| DINOv3 | `DINOv3Adapter`; [guide](../examples/configs/dinov3/README.md) | Preprocessed images |
| Pi0.5 | `Pi05Policy`; [guide](../examples/configs/pi05/README.md) | Checkpoint-compatible observations; optional explicit action noise |
| Wall-OSS | `WallOssPolicy`; [base](../examples/configs/wall_oss/README.md), [RTC](../examples/configs/wall_oss/rtc/README.md), [sorting](../examples/configs/wall_oss/sorting/README.md) | Images, instruction, proprioception and profile masks |
| HY Embodied / HyVLA | `HyEmbodiedPolicy` or component adapter; [UMI](../examples/configs/hy_embodied/README.md), [HyVLA](../examples/configs/hy_vla/README.md) | Preprocessed camera, text and state tensors |
| GR00T | Dedicated adapter/runner; [guide](../examples/configs/gr00t/README.md) | Checkpoint-compatible observation batch and backbone assets |
| G0.5 | `patch_g05_policy_for_rpu`; [guide](../examples/configs/g05/README.md) | Prepared camera/text/state input bundle and external policy source |
| InternVLA / NavDP | Full-chain runner and component adapters; [guide](../examples/configs/internvla/README.md) | Paired checkpoint bundle, CPU fixture and asset manifest for System2 → RGBD → NavDP |
| LingBot v1 / v2 | Component adapter or `build_lingbot_vla_v2`; [v1](../examples/configs/lingbot/README.md), [v2](../examples/configs/lingbot/v2/README.md) | v1: preprocessed observations; v2: checkpoint plus training YAML and camera/state/language tensors |
| RhinoVLA | `RhinoVLAPolicy`; [guide](../examples/configs/rhinovla/README.md) | Model-repository preprocessing and runtime factory |
| HALO | Action Expert runner; [guide](../examples/configs/halo/README.md) | Paired EMA checkpoint and CPU single-step fixture with prepared prefix K/V |

Core allocation, storage padding and cache construction are described in
[compute-core topology](planner_core_topology.md). Adapter ownership and shared
runtime contracts are indexed in the [implementation map](../knowledge/synthesis/model-family-map.md).

## Qwen3 speech

`Qwen3ASRModel` and `Qwen3TTSModel` provide dedicated ASR and TTS entry points
with CPU and device components. Use the [speech guide](qwen3_audio.md) for pinned
assets, input limits, source preparation and process ownership. Use the [ASR](../examples/configs/qwen3_asr/README.md) and
[TTS](../examples/configs/qwen3_tts/README.md) TOMLs with `examples.run`;
the dedicated speech APIs remain separate from the CausalLM/VLM registry.

## Whisper ASR

`WhisperASRModel` supports the pinned `csukuangfj/sherpa-onnx-whisper-base`
export. CPU work is limited to audio loading, resampling, log-mel features,
greedy token choice and text decoding; encoder, decoder, cross-attention, KV
cache and logits run on RPU. The fixed profile is FP16, batch 1, automatic
language detection and at most 30 minutes of audio. Host-side segmentation
also provides one-second streaming updates with revisable partial text and
confirmation after one second of silence. Timestamps, beam search and batched
inference are unsupported. See the
[Whisper example](../examples/configs/whisper_asr/README.md).
