# Whisper base ASR

## 配置说明

[base/fp16.toml](base/fp16.toml) 使用独立 `WhisperASRModel` API，在 RPU 上运行
`csukuangfj/sherpa-onnx-whisper-base` 的 encoder、decoder、KV cache 和 logits。
CPU 负责读取音频、重采样、log-mel、贪心选词和文本解码。

输入是单声道 PCM16 WAV，单次最长 30 分钟；模型自动判断语言。30 秒以内保持
原单段路径，长音频由 CPU 切分后复用同一组 RPU Graph。固定 batch 1、FP16、
greedy，单段文本上下文共 448 个位置（4 个 prompt，最多 444 个内容 token）。
不支持时间戳、Beam Search 或批量推理。

流式调用每累计 1 秒返回一次结果，连续静音 1 秒后把当前句子从临时文字移入
已确认文字：

```python
stream = model.create_stream(sample_rate=16000)
for update in stream.accept_audio(audio_chunk):
    text = update.confirmed_text + update.partial_text
final = stream.finish()
```

`confirmed_text` 后续不再变化，`partial_text` 在句子确认前允许修正。语言在首句
确认后固定。连续讲话达到 28 秒时会执行技术分段，并用 2 秒重叠 token 去重。
静音判断采用 -40 dBFS 固定门限，不包含降噪或通用 VAD。

<a id="quantization"></a>
## 准备与运行

checkpoint 目录必须包含固定版本的 `base-encoder.onnx`、`base-decoder.onnx` 和
`base-tokens.txt`。复制模板并填写 checkpoint、音频和新的输出目录，然后从仓库根目录执行：

```bash
python -m examples.run --config /path/to/whisper.toml --check-config
python -m examples.run --config /path/to/whisper.toml
```

相对路径以 TOML 所在目录为基准。固定协议执行 2 次预热和 10 次测量；这里只记录性能，
不设置业务性能门槛。

## 结果说明

`transcription.json` 保存文本、自动识别语言、token IDs 和结束原因。`report.json`
保存每次总耗时、CPU 时间、RTF、token/s、组件 P50/P95、模型加载耗时和 Graph
状态。输出目录必须为空，已有结果不会被覆盖。
