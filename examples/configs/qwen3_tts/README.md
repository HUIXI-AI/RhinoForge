# Qwen3 TTS Base 配置

<a id="qwen3-tts-配置"></a>

[配置总览](../README.md) · [全部配置](#configurations) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

输入中文或英文文本及 24 kHz 单声道参考音频，生成 WAV。文本长度为 1–1024 个字符，参考音频长 0.5–20 秒。当前 Base 入口只使用参考说话人嵌入，不提供 CustomVoice、VoiceDesign、ICL transcript 或流式输出。

## 选择模型、精度和 codec

<a id="配置矩阵"></a>

0.6B 与 1.7B Base 均提供 FP16、W8A16、W4A16 G32 和 NVFP4 + FP16 heads；0.6B 另有 CPU codec 配置。每份 TOML 列在[全部配置](#configurations)。

`model.codec_backend` 选择 codec 后段路线：默认 `rpu` 将解码使用的 31 层 Conv1d、6 层转置卷积（含重叠相加与 bias）、2 个 ConvNeXt 块及 SnakeBeta/残差单元卸载到 RPU FP16/ACC32（`cpu` 恢复声码器 CPU FP32 卷积与激活）；Talker、Predictor、codec Transformer（1–255 帧）和声码器各自的执行范围见具体 TOML 与[语音指南](../../../docs/qwen3_audio.md)。`rpu_execution.prefill.linear_acc32` 控制解码器和输出头的累加模式。

<a id="输入与核心"></a>

模板默认使用 8 核，可按登记配置选择 4/6/8 核；4/6 核预算下音频编码器、codec Transformer 和设备声码器卷积按 4 核配方执行。每项配置的输入限制见[全部配置](#configurations)。

<a id="运行"></a>

## 准备参考音频并运行

按[权重准备](../WEIGHTS.md)准备模型及配套源码，填写 `model.checkpoint`、`model.source`、`input.reference_audio` 和 `run.output_dir`。在 `input.text` 中写待合成文本，`input.language` 与文本语言对应。

从仓库根目录执行：

```bash
python -m examples.run --config /path/to/tts.toml --check-config
python -m examples.run --config /path/to/tts.toml
```

`--check-config` 只检查配置和输入参数范围；参考音频需要在运行前准备好。生成上限和 codec 参数见[参数说明](../PARAMETERS.md)，WAV 及其他输出见[结果说明](../../RESULTS.md)。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：Qwen3 TTS。

### Base 语音合成 · 0.6B

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 |
|---|---|---|---|---|---|---|
| [0_6b/fp16.toml](0_6b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测 | ● 已实测 | ● 已实测 |
| [0_6b/fp16_cpu_codec.toml](0_6b/fp16_cpu_codec.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测 | ● 已实测 | ● 已实测 |
| [0_6b/nvfp4.toml](0_6b/nvfp4.toml) | 支持（`SUPPORTED`） | nvfp4 | Talker / Predictor NVFP4；输出头 FP16 | ● 已实测 | ● 已实测 | ● 已实测 |
| [0_6b/w4a16_g32.toml](0_6b/w4a16_g32.toml) | 支持（`SUPPORTED`） | w4a16 | Talker / Predictor / 输出头 W4A16 G32 | ● 已实测 | ● 已实测 | ● 已实测 |
| [0_6b/w8a16.toml](0_6b/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Talker / Predictor / 输出头 W8A16 | ● 已实测 | ● 已实测 | ● 已实测 |

Python API 同样默认启用 `codec_backend="rpu"`；传入 `codec_backend="cpu"` 可切回 CPU：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 + 参考音频 |
| text 默认示例 | 你好，欢迎使用语音助手。 |
| language 默认值 | Chinese |
| max_new_tokens 默认值 | 256 |
| 文本可调范围 | 1–1024字符；Chinese / English |
| 图像/音频可调范围 | 单声道24 kHz；12000–480000采样点（0.5–20秒）；有限且非静音 |
| 生成可调范围 | 每段 max_new_tokens=2–256 |
| 其他输入形状 | 单条请求 |
| 联动限制 | 短路径≤128字符且≤32 token；长文本按 tokenizer 分段。每段需在生成预算内得到 EOS；仅 Base，不含 CustomVoice / VoiceDesign。 |
| 默认核数 | 8 |
| CPU线程数 | 16 |
| 权重 | `/path/to/Qwen3-TTS-12Hz-0.6B-Base`；准备方法见[权重准备](../WEIGHTS.md) |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/qwen3_audio/runner.py) · [qwen3_tts.py](../../../python/rpu_backend/api/qwen3_tts.py)。

### Base 语音合成 · 1.7B

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 |
|---|---|---|---|---|---|---|
| [1_7b/fp16.toml](1_7b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测 | ● 已实测 | ● 已实测 |
| [1_7b/nvfp4.toml](1_7b/nvfp4.toml) | 支持（`SUPPORTED`） | nvfp4 | Talker / Predictor NVFP4；输出头 FP16 | ● 已实测 | ● 已实测 | ● 已实测 |
| [1_7b/w4a16_g32.toml](1_7b/w4a16_g32.toml) | 支持（`SUPPORTED`） | w4a16 | Talker / Predictor / 输出头 W4A16 G32 | ● 已实测 | ● 已实测 | ● 已实测 |
| [1_7b/w8a16.toml](1_7b/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Talker / Predictor / 输出头 W8A16 | ● 已实测 | ● 已实测 | ● 已实测 |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 + 参考音频 |
| text 默认示例 | 你好，欢迎使用语音助手。 |
| language 默认值 | Chinese |
| max_new_tokens 默认值 | 256 |
| 文本可调范围 | 1–1024字符；Chinese / English |
| 图像/音频可调范围 | 单声道24 kHz；12000–480000采样点（0.5–20秒）；有限且非静音 |
| 生成可调范围 | 每段 max_new_tokens=2–256 |
| 其他输入形状 | 单条请求 |
| 联动限制 | 短路径≤128字符且≤32 token；长文本按 tokenizer 分段。每段需在生成预算内得到 EOS；仅 Base，不含 CustomVoice / VoiceDesign。 |
| 默认核数 | 8 |
| CPU线程数 | 16 |
| 权重 | `/path/to/Qwen3-TTS-12Hz-1.7B-Base`；准备方法见[权重准备](../WEIGHTS.md) |
| 入口 | [run.py](../../run.py) |

## 结果说明

输出目录必须为空（本次日志除外）。`output.wav` 为最后一次测量的 24 kHz FLOAT 音频；
`report.json` 包含各次整句耗时、分段耗时（`stage_ms_runs`：`speaker_embedding`、`talker`、`predictor`、`codec_decode`、`other`）、输出时长、RTF、CPU 时间及加载耗时。
RTF = 整句耗时 / 实际音频时长；正式计时排除加载、预热及文件读写（API 内参考音频读取仍计时）。
`cpu_ms_runs` 是进程所有线程累计 CPU 时间，可大于墙钟时间，不是 CPU 占用百分比。
`--profile` 另加一次诊断请求，不计入性能样本。
