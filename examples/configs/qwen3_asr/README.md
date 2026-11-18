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

检查通过会输出 `configuration OK`，只说明语法和输入参数范围符合要求。实际运行输出转写文本和 token 序列，字段见[结果说明](../../RESULTS.md)。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：Qwen3 ASR。

### 语音识别 · 0.6B

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 |
|---|---|---|---|---|---|---|
| [0_6b/fp16.toml](0_6b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测 | ● 已实测 | ● 已实测 |
| [0_6b/nvfp4.toml](0_6b/nvfp4.toml) | 支持（`SUPPORTED`） | nvfp4 | 文本解码器 NVFP4；输出头 / 音频编码器 FP16 | ● 已实测 | ● 已实测 | ● 已实测 |
| [0_6b/w4a16_g32.toml](0_6b/w4a16_g32.toml) | 支持（`SUPPORTED`） | w4a16 | 文本解码器 / 输出头 W4A16 G32；音频编码器 FP16 | ● 已实测 | ● 已实测 | ● 已实测 |
| [0_6b/w8a16.toml](0_6b/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | 文本解码器 / 输出头 W8A16；音频编码器 FP16 | ● 已实测 | ● 已实测 | ● 已实测 |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 单声道音频 |
| max_new_tokens 默认值 | 64 |
| 文本可调范围 | 模型构造转写提示词 |
| 图像/音频可调范围 | 16 kHz；8000–128000采样点（0.5–8秒）；有限浮点；audio_seconds=0.5–8 时取录音开头该秒数（录音不短于它，不补静音） |
| 生成可调范围 | max_new_tokens=1–64 |
| 其他输入形状 | B=1；无 padding |
| 联动限制 | 采样率和声道数必须匹配；音频占位数量等于编码长度，prompt token数+64不得超过cache。 |
| 默认核数 | 8 |
| CPU线程数 | 4 |
| 权重 | `/path/to/Qwen3-ASR-0.6B`；准备方法见[权重准备](../WEIGHTS.md) |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/qwen3_audio/runner.py) · [qwen3_asr.py](../../../python/rpu_backend/api/qwen3_asr.py)。

### 语音识别 · 1.7B

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 |
|---|---|---|---|---|---|---|
| [1_7b/fp16.toml](1_7b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测 | ● 已实测 | ● 已实测 |
| [1_7b/nvfp4.toml](1_7b/nvfp4.toml) | 支持（`SUPPORTED`） | nvfp4 | 文本解码器 NVFP4；输出头 / 音频编码器 FP16 | ● 已实测 | ● 已实测 | ● 已实测 |
| [1_7b/w4a16_g32.toml](1_7b/w4a16_g32.toml) | 支持（`SUPPORTED`） | w4a16 | 文本解码器 / 输出头 W4A16 G32；音频编码器 FP16 | ● 已实测 | ● 已实测 | ● 已实测 |
| [1_7b/w8a16.toml](1_7b/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | 文本解码器 / 输出头 W8A16；音频编码器 FP16 | ● 已实测 | ● 已实测 | ● 已实测 |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 单声道音频 |
| max_new_tokens 默认值 | 64 |
| 文本可调范围 | 模型构造转写提示词 |
| 图像/音频可调范围 | 16 kHz；8000–128000采样点（0.5–8秒）；有限浮点；audio_seconds=0.5–8 时取录音开头该秒数（录音不短于它，不补静音） |
| 生成可调范围 | max_new_tokens=1–64 |
| 其他输入形状 | B=1；无 padding |
| 联动限制 | 采样率和声道数必须匹配；音频占位数量等于编码长度，prompt token数+64不得超过cache。 |
| 默认核数 | 8 |
| CPU线程数 | 4 |
| 权重 | `/path/to/Qwen3-ASR-1.7B`；准备方法见[权重准备](../WEIGHTS.md) |
| 入口 | [run.py](../../run.py) |

## 结果说明

输出目录必须为空（本次日志除外），已有结果不会覆盖。
`transcription.json` 保存文本、原始文本和 token IDs；`report.json` 包含每次耗时、
分段耗时（`stage_ms_runs`：`feature_extract`、`audio_encoder`、`prefill`、`decode`、`other`）、
中位数、模型加载耗时、Graph 统计及 RTF。ASR RTF 为转写耗时除以输入音频时长。
`cpu_ms_runs` 是进程所有线程累计 CPU 时间，不是 CPU 占用百分比。
`--profile` 在正式计时后额外采集一次 Torch profile。运行结果不自动认证识别准确率。
