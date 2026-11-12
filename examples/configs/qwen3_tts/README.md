# Qwen3-TTS 12Hz 0.6B Base

## 配置说明

| 模板 | 声码器后段 |
|---|---|
| [0_6b/base.toml](0_6b/base.toml) | 默认：31 层解码 Conv1d 与 6 层转置卷积投影使用 RPU FP16/ACC32，剩余激活为 CPU |
| [0_6b/base_rpu_codec.toml](0_6b/base_rpu_codec.toml) | 显式选择同一 RPU 配方，保留既有配置路径 |

两者均复用设备 Talker、Predictor 及现有 codec Transformer 路由。
省略 `codec_backend` 时也默认使用 RPU；如需原 FP32 Conv1d，将该字段改为 `"cpu"`。
RPU 卷积路径仍为经过有限样本验证的实验性实现。
只接受固定 Base checkpoint、中英文文本 1–1024 字符和单声道 24 kHz、0.5–20 秒非静音参考音频。
不是完整声码器全 RPU 迁移。详细采样、分段和前段 CPU 回退范围见[语音指南](../../../docs/qwen3_audio.zh.md)。

<a id="quantization"></a>
## 准备与运行

按[语音指南](../../../docs/qwen3_audio.zh.md)安装 audio 依赖、匹配运行库、固定版本权重及准备后的源码。
复制模板，填写 `checkpoint`、`source`（准备后的 `qwen_tts/` 目录）、`input.reference_audio`、
`input.text` 和新的 `run.output_dir`，从仓库根目录执行：

```bash
python -m examples.run --config /path/to/tts.toml --check-config
python -m examples.run --config /path/to/tts.toml
```

相对资产和输出路径以 TOML 所在目录为基准。配置检查允许占位路径，不加载 Torch 或设备。
默认 16 个 CPU 线程，2 次预热后测 5 次；模型只加载一次，每次合成前重置 `run.seed`。
在空闲设备上按机器约定获取设备锁，ASR/TTS 分进程运行。

Python API 同样默认启用；传入 `codec_backend="cpu"` 可切回 CPU：

```python
with Qwen3TTSModel.from_pretrained(checkpoint, source=source) as model:
    waveform, rate = model.synthesize(text, reference_audio=reference)
```

## 结果说明

输出目录必须为空（本次日志除外）。`output.wav` 为最后一次测量的 24 kHz FLOAT 音频；
`report.json` 包含各次整句耗时、输出时长、RTF、CPU 时间及加载耗时。
RTF = 整句耗时 / 实际音频时长；正式计时排除加载、预热及文件读写（API 内参考音频读取仍计时）。
`cpu_ms_runs` 是进程所有线程累计 CPU 时间，可大于墙钟时间，不是 CPU 占用百分比。
`--profile` 另加一次诊断请求，不计入性能样本。

`execution_stats[].codec_convolutions` 记录后段选择、31 个已安装层和累计 RPU 调用次数。
RPU 模式会改变波形舍入，需比较同 tokens 的 CPU 基准并核验内容；不能用运行成功替代音质检查。
它保留 CPU 补边、列展开、bias、SnakeBeta、转置卷积重叠相加和上采样整理，不保证所有文本都更快。
