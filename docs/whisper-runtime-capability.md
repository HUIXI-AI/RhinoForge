# Whisper base RhinoForge 运行能力门禁

## 结论

`csukuangfj/sherpa-onnx-whisper-base` 可以进入 RhinoForge 实现阶段。R1 运行时
具备 FP16 Whisper base 所需的 Conv1d、Linear、LayerNorm、ERF GELU、self/cross
attention 和 KV 写入能力。当前结论只代表底层能力通过，不代表端到端推理已经完成。

## 锁定范围

- 上游 revision：`bb53ee204431c90d314c1cc08d28d23e5b7927cc`
- 资产：`base-encoder.onnx`、`base-decoder.onnx`、`base-tokens.txt`
- 固定 profile：batch 1、FP16、16 kHz 单声道、最长 30 秒、448 text context
- 资产大小、SHA256、模型结构和参考转录见
  `tests/fixtures/whisper_base/manifest.json`

## R1 实测

测试环境：`agi@10.10.66.42`，RhinoForge runtime v1.2.0。第 0 层固定
prefill3/decode1 产物在 2026-10-21 重新运行，结果如下：

| 路径 | hidden cosine | K cosine | V cosine |
| --- | ---: | ---: | ---: |
| prefill3 | 0.999999645 | 0.999999837 | 0.999999815 |
| decode1 | 0.999999691 | 0.999999939 | 0.999999937 |

单算子探针最低结果：LayerNorm 0.999999523、ERF GELU 0.999999881、
Linear 0.999998391、cross attention 0.999999821、self attention 1.0、
Conv1d 0.999996781，均通过 0.9999 门槛。

## 实现约束

通用 `F.conv1d` Graph 虽然没有 CPU callback，且 BUILD/REPLAY 计数稳定，但
A/B 输入刷新测试失败。因此 Whisper encoder 不复用该 Graph 封装，必须由独立
native owner 管理持久 buffer、输入 DMA、执行和释放，并增加 A/B/A 刷新回归测试。

R1 的 `/mnt/ssd_data` 当前已满。临时真机产物应写入 `/dev/shm`，不得清理或覆盖
其他用户数据。
