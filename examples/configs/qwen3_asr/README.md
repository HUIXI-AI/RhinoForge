# Qwen3-ASR 0.6B / 1.7B

## 配置说明

| 模板 | 模型 | 精度配方 |
|---|---|---|
| [0_6b/fp16.toml](0_6b/fp16.toml) | Qwen3-ASR-0.6B | 文本解码器与输出头 FP16 |
| [0_6b/w8a16.toml](0_6b/w8a16.toml) | Qwen3-ASR-0.6B | 文本解码器与输出头 W8A16 |
| [0_6b/w4a16_g32.toml](0_6b/w4a16_g32.toml) | Qwen3-ASR-0.6B | 文本解码器与输出头 W4A16 (G32) |
| [0_6b/nvfp4.toml](0_6b/nvfp4.toml) | Qwen3-ASR-0.6B | 文本解码器 NVFP4；输出头 FP16 |
| [1_7b/fp16.toml](1_7b/fp16.toml) | Qwen3-ASR-1.7B | 文本解码器与输出头 FP16 |
| [1_7b/w8a16.toml](1_7b/w8a16.toml) | Qwen3-ASR-1.7B | 文本解码器与输出头 W8A16 |
| [1_7b/w4a16_g32.toml](1_7b/w4a16_g32.toml) | Qwen3-ASR-1.7B | 文本解码器与输出头 W4A16 (G32) |
| [1_7b/nvfp4.toml](1_7b/nvfp4.toml) | Qwen3-ASR-1.7B | 文本解码器 NVFP4；输出头 FP16 |

以上模板均使用独立 `Qwen3ASRModel` API，支持 `rpu_execution.model.num_cores = 4 / 6 / 8`（默认 8）与 `rpu_execution.prefill.linear_acc32`。音频编码器保持 FP16/ACC32。输入为单声道 16 kHz、0.5–8 秒本地音频；greedy 解码最多 64 tokens。

<a id="quantization"></a>
## 准备与运行

按[语音指南](../../../docs/qwen3_audio.zh.md)安装 audio 依赖和匹配运行库、下载固定版本 0.6B 或 1.7B 权重。
复制模板，填写 `checkpoint`、`input.audio` 和新的 `run.output_dir`，从仓库根目录执行：

```bash
python -m examples.run --config /path/to/asr.toml --check-config
python -m examples.run --config /path/to/asr.toml
```

相对资产和输出路径以 TOML 所在目录为基准。配置检查不加载 Torch、设备或权重，允许占位路径；
实际执行会检查音频。默认 4 个 CPU 线程，2 次预热后测 5 次。ASR/TTS 分进程运行，
运行前按机器约定获取设备锁。

## 结果说明

输出目录必须为空（本次日志除外），已有结果不会覆盖。
`transcription.json` 保存文本、原始文本和 token IDs；`report.json` 包含每次耗时、
分段耗时（`stage_ms_runs`：`feature_extract`、`audio_encoder`、`prefill`、`decode`、`other`）、
中位数、模型加载耗时、Graph 统计及 RTF。ASR RTF 为转写耗时除以输入音频时长。
`cpu_ms_runs` 是进程所有线程累计 CPU 时间，不是 CPU 占用百分比。
`--profile` 在正式计时后额外采集一次 Torch profile。运行结果不自动认证识别准确率。
