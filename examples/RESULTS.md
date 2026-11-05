# 日志与结果

[模型示例](README.md) · [通用参数](configs/PARAMETERS.md)

<a id="logging-and-performance-output"></a>

## 日志设置

在复制的 TOML 中加入以下顶层配置表，与 `[run]` 并列。省略 `[logging]` 时也使用这组默认值：

```toml
[logging]
level = "ERROR"
summary = true
```

默认终端显示 `[MODEL]` 模型配置摘要、`[PERF]` 本次运行摘要，以及实际错误。模型摘要展示模板中填写的模型、精度、输入、执行和采样设置；省略字段的实际线程、布局等信息按运行报告查看。普通加载信息、WARNING/INFO 级的 fallback/Graph 信息、生成文本和完整 JSON 报告默认隐藏。

### 日志等级对照

`level` 必须使用下表中的大写值。它只过滤诊断信息；`summary = true` 时，五个等级都会显示 `[MODEL]` 和 `[PERF]`。

| `level` | 显示的诊断等级 | 典型内容与用途 |
|---|---|---|
| `CRITICAL` | 仅 CRITICAL | 仅保留致命错误；普通 ERROR 也会隐藏 |
| `ERROR`（默认） | ERROR、CRITICAL | 显示实际错误；显示模型配置和运行摘要 |
| `WARNING` | WARNING、ERROR、CRITICAL | 增加告警，包括相应等级的 CPU fallback、Graph 信息 |
| `INFO` | INFO 及更严重等级 | 增加加载过程、普通 `print()` 输出和生成文本 |
| `DEBUG` | DEBUG 及更严重等级 | 增加已有 Graph 调试信息和完整报告；原生 TRACE 归入 DEBUG |

没有错误时，`ERROR` 与 `CRITICAL` 的正常输出相同。普通 `print()` 默认归入 INFO；带有可识别等级标签的输出按其标签过滤。Python 异常仍会传播，不会因日志等级而被吞掉。

下面用模拟日志展示各等级的输出内容；模型与运行摘要由 `summary` 单独控制。

```text
# WARNING 开始显示
[WARNING] DEMO CPU fallback diagnostic（模拟告警）
[WARNING] [RpuKernelGraph] WARN: DEMO native Graph diagnostic（模拟告警）
# INFO 在上述基础上增加
[INFO] DEMO loading checkpoint（模拟普通 print）
[INFO] DEMO generated text: <模型生成文本>
# DEBUG 再增加
[DEBUG] DEMO Graph BUILD/REPLAY diagnostic（模拟调试行）
[DEBUG] report={...完整报告...}
[DEBUG] [TRACE] DEMO native trace diagnostic（模拟调试行）
```

修改同一 TOML 中的 `level` 后，运行命令保持不变：

```bash
python examples/run.py --config /tmp/pi05.toml
```

`summary = false` 只关闭 `[MODEL]` 和 `[PERF]`，不改变诊断等级过滤。没有 `[logging].file` 字段。原生 stdout/stderr 保留在 `run.output_dir` 下的 `native-*.stdout.log`、`native-*.stderr.log`；运行结束时按所选级别回放，因此可能出现在运行摘要之后；SDK 原始输出不会直接刷终端。完整 `report.json` 的保存不受终端日志级别影响。

## 输出文件

`run.output_dir` 指定输出位置；runner 可以在其中创建子目录。`report.json` 记录实际配置、输入身份和产物路径；动作、特征和生成结果的文件名见模型页。文件保存不受终端日志级别影响。
