# Qwen3-TTS 12Hz 0.6B / 1.7B Base

## 配置说明

| 模板 | 模型 | 精度配方 | 声码器残差卷积 |
|---|---|---|---|
| [0_6b/fp16.toml](0_6b/fp16.toml) | Qwen3-TTS-12Hz-0.6B-Base | Talker / Predictor 及输出头 FP16 | 默认 `rpu`（12 层残差卷积走 RPU 紧凑 slab） |
| [0_6b/fp16_cpu_codec.toml](0_6b/fp16_cpu_codec.toml) | Qwen3-TTS-12Hz-0.6B-Base | Talker / Predictor 及输出头 FP16 | `cpu`（残差卷积回退 CPU FP32） |
| [0_6b/w8a16.toml](0_6b/w8a16.toml) | Qwen3-TTS-12Hz-0.6B-Base | Talker / Predictor 及输出头 W8A16 | `rpu` |
| [0_6b/w4a16_g32.toml](0_6b/w4a16_g32.toml) | Qwen3-TTS-12Hz-0.6B-Base | Talker / Predictor 及输出头 W4A16 (G32) | `rpu` |
| [0_6b/nvfp4.toml](0_6b/nvfp4.toml) | Qwen3-TTS-12Hz-0.6B-Base | Talker / Predictor NVFP4；输出头 FP16 | `rpu` |
| [1_7b/fp16.toml](1_7b/fp16.toml) | Qwen3-TTS-12Hz-1.7B-Base | Talker / Predictor 及输出头 FP16 | `rpu` |
| [1_7b/w8a16.toml](1_7b/w8a16.toml) | Qwen3-TTS-12Hz-1.7B-Base | Talker / Predictor 及输出头 W8A16 | `rpu` |
| [1_7b/w4a16_g32.toml](1_7b/w4a16_g32.toml) | Qwen3-TTS-12Hz-1.7B-Base | Talker / Predictor 及输出头 W4A16 (G32) | `rpu` |
| [1_7b/nvfp4.toml](1_7b/nvfp4.toml) | Qwen3-TTS-12Hz-1.7B-Base | Talker / Predictor NVFP4；输出头 FP16 | `rpu` |

以上模板均复用设备 Talker、Predictor 及 codec Transformer 路由，支持 `rpu_execution.model.num_cores = 4 / 6 / 8`（默认 8）与 `rpu_execution.prefill.linear_acc32`。
省略 `codec_backend` 时默认使用 `rpu`；如需原 CPU FP32 残差卷积，将该字段改为 `"cpu"`。
只接受固定 Base checkpoint、中英文文本 1–1024 字符和单声道 24 kHz、0.5–20 秒非静音参考音频。
详细采样、分段流式生成（`stream_synthesize`）、分段耗时和前段 CPU 回退范围见[语音指南](../../../docs/qwen3_audio.zh.md)。

<a id="quantization"></a>
## 准备与运行

按[语音指南](../../../docs/qwen3_audio.zh.md)安装 audio 依赖、匹配运行库、固定版本 0.6B / 1.7B 权重及准备后的源码。
复制模板，填写 `checkpoint`、`source`（准备后的 `qwen_tts/` 目录）、`input.reference_audio`、
`input.text` 和新的 `run.output_dir`，从仓库根目录执行：

```bash
python -m examples.run --config /path/to/tts.toml --check-config
python -m examples.run --config /path/to/tts.toml
```

相对资产和输出路径以 TOML 所在目录为基准。配置检查允许占位路径，不加载 Torch 或设备。
默认 16 个 CPU 线程，2 次预热后测 5 次；模型只加载一次，每次合成前重置 `run.seed`。
在空闲设备上按机器约定获取设备锁，ASR/TTS 分进程运行。

Python API 同样默认启用 `codec_backend="rpu"`；传入 `codec_backend="cpu"` 可切回 CPU：

```python
with Qwen3TTSModel.from_pretrained(checkpoint, source=source) as model:
    waveform, rate = model.synthesize(text, reference_audio=reference)
```

## 结果说明

输出目录必须为空（本次日志除外）。`output.wav` 为最后一次测量的 24 kHz FLOAT 音频；
`report.json` 包含各次整句耗时、分段耗时（`stage_ms_runs`：`speaker_embedding`、`talker`、`predictor`、`codec_decode`、`other`）、输出时长、RTF、CPU 时间及加载耗时。
RTF = 整句耗时 / 实际音频时长；正式计时排除加载、预热及文件读写（API 内参考音频读取仍计时）。
`cpu_ms_runs` 是进程所有线程累计 CPU 时间，可大于墙钟时间，不是 CPU 占用百分比。
`--profile` 另加一次诊断请求，不计入性能样本。
