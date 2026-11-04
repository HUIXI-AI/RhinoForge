# 模型运行与调试

**简体中文** | [English](model_testing.md)

从[模型示例](../examples/README.zh.md)选择 TOML。运行器位于 `examples/runners/`，共享配置与执行代码位于 `examples/common/`。

## 运行

```sh
python -m examples.run --config examples/configs/qwen3/text/0_6b/fp16.toml --check-config
python -m examples.run --config examples/configs/qwen3_vl/vl/4b/w8a16_text_head_vision.toml
python -m examples.run --config examples/configs/pi05/2cam/w8a16_action_nvfp4.toml
```

按模型指南填写 checkpoint 和输入，替换占位路径。`input.prompt_file` 和 `input.prepared_input_safetensors` 的相对路径按配置文件所在目录解析；其他资产路径按模型指南填写。`--check-config` 检查配置，不加载权重或访问设备。量化权重须携带加载器要求的元数据，见[量化](quantization.zh.md)。

预处理输入使用对应 checkpoint 的 processor 产生的 CPU 张量字典，形状与组件须符合所选模型要求。

## 执行参数

- `run.warmup` 默认两次调用，让新 Graph 完成 BUILD 和 REPLAY；`run.runs` 设置后续调用次数。
- `warmup >= 2` 时，省略 `warmup_decode_steps` 或令其等于 `input.decode_steps`。显式 `warmup=0/1` 可做缩短 decode 的冷路径诊断。
- decode 步数不含 prefill 产生的首 token。EOS 可提前结束；固定长度运行时，两阶段均设 `stop_on_eos=false`。
- `run.output_dir` 指定输出目录；在生成文件中查看输出和执行状态。是否提供 Graph 统计取决于 runner，普通 Qwen3 报告不采集这些统计；仅有 BUILD/REPLAY 日志不能证明 Graph 不变量成立。

## 诊断调用

`--profile` 或 `[run].profile=true` 在常规运行后增加一次 Torch profiler 调用。按需启用 shape、memory 或 stack。跟踪可能包含应用数据，保存在源码仓库外。

Qwen3 公开 runner 支持 `--hwperf` 或 `[run].hwperf=true`，需要匹配的 sanitized
Release Launch。在空闲且已持有平台设备锁的板卡上可运行：

```bash
python -m examples.run --config qwen3.local.toml --hwperf --check-config
python -m examples.run --config qwen3.local.toml --hwperf
```

配置检查本身不采集或验证 trace。HWPerf 在模型准备及 Graph BUILD 前启用，硬件
trace 写入本次结果的 `hwperf/` 目录；请求后未生成 trace 或无法解析时执行失败。
`[runner.hw_profile].max_dumps` 限制 dump 数，默认 32。硬件跟踪影响执行窗口，
不能将其耗时与未跟踪运行直接比较。支持情况取决于 runner：Wall-OSS base
延迟入口仍拒绝这些跟踪开关。trace 应保存在仓库外，并记录源码、运行资产和输入
shape。详见[运行时配置](runtime_config.zh.md#profiling-机制)和[模型验证](validation_policy.zh.md)。

## 解读 Qwen3 首次运行

[首次运行指南](getting_started.zh.md#4-解读短-case-结果)使用固定 Qwen3-0.6B 权重、
裸 P7 提示词、D4、W2/R3 和 FP16 计算。D4 含 prefill 首 token 时最多生成 5 token，
没有应用 chat template。默认结果父目录是 `perf_results/examples`；可将
`run.output_dir` 设为仓库外的任务目录。每次 Qwen3 执行在独立 profile 子目录写入
`report.json` 和最后一轮 `output.pt`。`PASS` / `execution.status=COMPLETED`
表示执行完成，仍明确记录 `correctness=not_run`、`formal_certification=false`。
数值结论需要另行执行语义一致的参考检查。

## Qwen3 语音

安装 `audio` 可选依赖，调用 `Qwen3ASRModel` 或 `Qwen3TTSModel`。
下载权重、准备 TTS 源码及完整 Python 用法见[语音指南](qwen3_audio.zh.md)。

语音也可通过 `python -m examples.run --config examples/configs/qwen3_tts/0_6b/base.toml --check-config` 检查，填写本地路径后执行。[ASR](../examples/configs/qwen3_asr/README.md) / [TTS](../examples/configs/qwen3_tts/README.md) 入口记录常驻推理耗时、累计 CPU 时间和 RTF。

## 仓库检查

运行 `make ci-local-core PYTHON=python` 执行主机测试，测试范围和设备用例排除清单与 CI 一致。
`make ci-local-quickstart` 检查示例配置和适配器注册表。

按[安装指南](getting_started.md)配置公开 SDK 和源码构建后，可在同一构建目录启用 C++ 契约测试：

```sh
cmake -S . -B build -DRHINOFORGE_BUILD_HOST_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
cmake -S . -B build -DRHINOFORGE_BUILD_DEVICE_TESTS=ON
cmake --build build --target rhino_device_tests
```

设备测试程序位于 `build/tests/`，配置运行资产后在空闲设备上执行。
两个选项默认关闭；设备测试独立于主机 CI。
