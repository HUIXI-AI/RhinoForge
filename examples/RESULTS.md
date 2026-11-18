# 日志与结果

[模型示例](README.md) · [通用参数](configs/PARAMETERS.md) · 性能报告

运行结束后，先核对模型、输入和运行次数，再读取计时与输出文件。终端摘要用于快速查看；复现或比较结果时，以 `run.output_dir` 中保存的报告和原始输出为准。

本页仅解释日志和结果文件，不列性能或数值误差测量结果。

<a id="logging-and-performance-output"></a>

<a id="日志设置"></a>

## 日志设置

在复制的 TOML 中加入以下配置表，与 `[run]` 并列：

```toml
[logging]
level = "ERROR"
summary = true
```

`level` 过滤诊断信息，`summary` 单独控制 `[MODEL]`、`[PERF]` 等摘要。省略 `[logging]` 时采用上面的默认值。Python 异常仍会传播。

| `level` | 终端诊断信息 |
|---|---|
| `CRITICAL` | 只显示致命错误，普通 ERROR 也会被隐藏 |
| `ERROR` | 显示错误和致命错误；默认值 |
| `WARNING` | 增加告警 |
| `INFO` | 增加加载信息、普通 `print()` 输出和生成文本 |
| `DEBUG` | 增加已有调试信息；部分专用入口还会输出完整报告 |

日志级别不影响报告文件的保存。

原生 stdout/stderr 保存在结果目录的日志文件中，运行结束时按所选级别回放。因此，一部分诊断信息可能出现在性能摘要之后。正常 Graph BUILD/REPLAY 带 INFO 级别；stderr 有内容本身不表示执行失败。仍需检查异常、退出码和报告状态，不能把真实 ERROR 降为普通信息。

## 如何读取耗时

比较前要匹配输入、权重、核数、CPU 线程、累加精度和采样设置。预热与正式采样分开；缺少的指标按未测或未保存处理。

| 输出类型 | 阅读时核对什么 |
|---|---|
| LLM / VLM | 实际输入 token 数、decode 调用数、输出 token 数，以及报告采用的吞吐统计方法。Prefill 已产生首个输出 token，输出数可能比 decode 次数多 1 |
| VLA 完整推理 | E2E 的计时起止位置、预热与正式次数、采用均值还是中位数 |
| Vision / Action 等组件 | 组件覆盖的计算范围；组件耗时不能直接相加当作完整请求耗时 |

具体的 TTFT、TPOT 和误差定义见性能报告的指标口径。未单独测量的 Vision 耗时或 TTFT 不能从其他阶段推算。

## 输出文件

`run.output_dir` 指定结果目录。不同模型可能创建文本、音频、图像特征或动作结果文件，具体文件名以对应模型页和 TOML 为准。运行结束后先检查实际输入、输出形状和状态，再读取结果文件。

示例：

```bash
python examples/run.py --config <config.toml> --check-config
python examples/run.py --config <config.toml>
```

输入、权重、核数或累加选项发生变化后，重新检查配置并启动新进程。数值误差需要与同输入、同参考语义的 CPU 输出比较；普通运行输出不能直接继承历史报告的误差结论。
