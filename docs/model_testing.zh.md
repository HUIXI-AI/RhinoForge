# 模型执行与性能采集

简体中文 | [English](model_testing.md)

`examples/` 中的模型脚本提供直接入口；`examples/run_model.py` 为同一入口应用
TOML 中的环境设置和可选 Torch profiler。配置按模型归档，路径和精度语义见
[模型示例](../examples/README.md)。
[完整配置目录](../examples/configs/README.md) 包含 Qwen3.5 Text/VL、Qwen3-VL 独立组件，
以及精度、核数和输入长度组合。新增 `[example]/[input]/[run]` 格式（Pi 用 `[pi05]`）
仍通过同一模型入口运行。少数独立集成入口使用 `[runner]/[model]`，不再为同一模型保留两套模板；两种格式不要混用。

## 运行一个配置

```sh
python examples/run_model.py --config examples/configs/qwen3/text/0_6b/fp16.toml --check-config
python examples/run_model.py --config examples/configs/qwen3_vl/vl/4b/w8a16.toml
python examples/pi05.py --config examples/configs/pi05/2cam/w8a16_action_nvfp4.toml
```

`--check-config` 不加载模型或访问板卡。执行前替换占位路径；TOML 中模型、图像、
batch 和输出路径均相对命令工作目录解析。量化目录必须带有加载器要求的元数据，
更改目录名称不会转换权重。转换方法见[量化说明](quantization.zh.md)。

需要预处理 tensor 的示例使用调用者按公开 checkpoint processor 生成的 CPU 张量映射。
组件示例不代表完整机器人策略或应用，实际输入范围由运行时前验决定。
历史认证记录不作为当前实现的支持范围，也不会扩大示例的适用范围。

## Torch profile

Qwen 与策略模板使用 `[run].profile = true` 或 `--profile`。使用 `[runner/model/request]` 的独立集成模板则使用：

```toml
[runner.torch_profile]
enabled = true
output = "profiles/torch_trace.json"
record_shapes = false
profile_memory = false
with_stack = false
```

`[run].profile` 在计时样本结束后额外执行一次诊断推理。
显式设置 `run.warmup` 和 `run.runs`；短测可使用 1 次预热、2–3 个样本和
`input.decode_steps = 32`，并保持其他输入及执行配置一致，`warmup_decode_steps` 不超过该值。
decode 步数是 prefill 首 token 之后的调用数，EOS 可以提前终止。
长 decode 模板只在显式选择后运行；示意图片不能复现历史报告的性能输入。

Profiler 包围一次执行并会改变时延；性能数字应在关闭 profiler 后采集。
除非诊断需要，保持 shape、内存和调用栈采集关闭。trace 可能包含应用数据。
完整参数见[运行时配置](runtime_config.zh.md)。

## Hardware performance trace

公开示例 runner 在配置检查时拒绝 `hwperf=true`。
只有具体算子或同步问题未解决时才使用已安装 runtime 的硬件 profiler。
trace 与稳态时延分开采集，记录实际算子资产和输入形状；低层 trace 由应用方保管，
不进入源码仓库。

## 小范围检查

板测前检查空闲并持有设备锁，使用绑定目标 worktree 的隔离环境。
每条受影响路径选一个代表：通常回放 3–10 次，性能测试 2 次预热、5–10 次测量。
先确认计算路径，再记录普通累计数值误差。受影响检查通过后停止，见
[验证预算](validation_policy.zh.md)和[性能测量](performance.zh.md)。
