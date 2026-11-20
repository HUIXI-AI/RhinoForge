# LightNav-0 R1 运行指南

简体中文 | [English](lightnav_0_r1.md)

本适配面向公开模型 `LightOriginsHQ/LightNav-0`，固定使用：

- RhinoForge 分支：`feat/lightnav-0-r1-cache`
- LightNav-0 revision：`826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`
- LightNav 上游源码 commit：`3015508b70fb6e30bbd66701e94c9b916dcb8df8`
- R1 runtime：v1.2.0

模型每次输出 5 个 token：2 个指向 token 和 3 个 RVQ 动作 token。

## 交付配置

| 项目 | 配置 |
|---|---|
| 输入窗口 | 最多 9 帧历史，加 1 帧当前画面 |
| 视觉缓存 | 2 帧一个 tubelet；完整滑窗稳态为 4 命中、1 缺失 |
| 预取 | `prefetch_latest=False`，避免重复计算复制 tubelet |
| 文本精度 | 36 层的 252 个 Attention/MLP 投影全部 W8A16 |
| 视觉精度 | Vision Encoder W8A16，pooler/DeepStack 后处理保持 FP16 |
| 保留 FP16 | 动作输出 head、embedding 和 norm |
| 输出 head | 五位置动作语法，每个位置物理补齐为 1408 行 |
| Decode | 标准自回归生成 5 token |
| 上一请求 token 候选 | 关闭 |
| Graph | 完整窗口建立后安全重绑并复用 Prefill/Decode Graph |

视觉缓存保存每个 tubelet 的 pooler output 和三层 DeepStack 特征。历史命中时直接复用，
当前缺失的 tubelet 才执行 ViT。视觉 embedding 和 M-RoPE 随请求变化，因此文本侧仍对
当前请求执行完整 Prefill；Graph 复用只减少建图和内核调度开销，不复用上一请求的文本
结果。

Text W8 使用输入激活二阶矩校准，Vision W8 使用 1.0 编码器范围。五位置动作语法
head 使用随 checkpoint 固化的校准参数，输出仍为标准的 5 个动作 token。

## 环境准备

该 GitHub 分支只包含 RhinoForge 源码，不包含 runtime、预编译 `rpu_backend.so`、
LightNav 上游源码和模型权重。

1. 按[统一安装指南](getting_started.zh.md)创建 Python 3.12 环境，并从当前 checkout
   编译安装 RhinoForge。不要复用其他提交生成的 `rpu_backend.so`。
2. 按[运行时资产说明](runtime_assets.zh.md#配置路径)配置正式 v1.2.0 runtime，设置
   `RHINO_LAUNCH_ROOT`、`RPU_KERNEL_LIB_PATH`、
   `RPU_KERNEL_LIB_PATH_EXPANSION` 和 `LD_LIBRARY_PATH`。runtime 必须与
   [`release/runtime-dependencies.json`](../release/runtime-dependencies.json)匹配。
3. 单独准备本文开头指定版本的 LightNav 源码和完整模型。不要在 LightNav 目录执行
   `pip install -e .`；运行脚本通过 `--lightnav-source` 加载源码。

模型目录必须包含权重、`eval_config.json`、processor/tokenizer 文件和
`action_tokenizer/`。

## W8A16 权重转换

先在原始未量化 CPU 模型（按 LightNav 上游加载口径为 BF16）上采集 Text 36 层、7 类
投影的输入激活二阶矩。当前校准使用客户
35 帧序列中的第 9、14、19、24、29、34 帧完整窗口，并按 FP16 的 5 个输出 token
执行 teacher forcing。这里的帧号从 0 开始。统计文件必须恰好包含 252 个 `float32`
向量。

先用原始 FP16 checkpoint 按[运行真实图片序列](#运行真实图片序列)生成推理结果，再采集统计：

```bash
python examples/lightnav_collect_activation_stats.py \
  --input /path/to/dataset.json \
  --images-dir /path/to/images \
  --teacher-output /path/to/fp16_result.json \
  --checkpoint /path/to/LightNav-0 \
  --lightnav-source /path/to/LightNav-0-source \
  --history-frames 9 \
  --frame-indices 9,14,19,24,29,34 \
  --output /outside/the/repository/lightnav_activation_moments.safetensors
```

`--teacher-output` 必须是同一图片序列的 FP16 推理结果，每个指定帧必须恰好有 5 个
token。统计文件是外部模型资产，仓库的 `*.safetensors` 规则会忽略它；不要强制加入 Git。

从原始 FP16 checkpoint 生成全 Text W8 + Vision W8 配置：

```bash
PYTHONPATH=python python -m rpu_backend.quant.convert_qwen3_vl \
  --bits 8 \
  --lightnav-activation-stats /path/to/lightnav_activation_moments.safetensors \
  --lightnav-vision-w8 \
  --src /path/to/LightNav-0 \
  --dst /path/to/LightNav-0-activation-vision-w8a16
```

转换器对 252 个 Text 投影执行激活感知逐输出通道 W8 量化；Vision Encoder 在安装时
执行逐输出通道 W8 量化。转换器会拒绝缺失、多余、形状错误或非有限的激活统计，也只
接受精确匹配的 LightNav-0 配置。校准统计作为模型资产单独交付，不在 Git 仓库内。

## 运行真实图片序列

输入 JSON 必须包含非空的 `prompt`（或 `instruction`）和按时间排序的 `images` 数组；
仅支持 `task="tracking"`。图片路径默认相对于 JSON，也可用 `--images-dir` 指定。

运行命令：

```bash
python examples/lightnav_inference.py \
  --input /path/to/dataset.json \
  --images-dir /path/to/images \
  --checkpoint /path/to/LightNav-0-activation-vision-w8a16 \
  --lightnav-source /path/to/LightNav-0-source \
  --history-frames 9 \
  --constrained-head \
  --output /path/to/inference_result.json
```

输出 JSON 按帧记录 5 个 token、输入 token 数、视觉缓存命中/缺失和各阶段耗时。
`total` 包含准备、Vision 和文本生成；`frame_total` 还包含 Graph 清理和请求外围处理，
两者都不包含模型加载和全部图片的初始解码/缩放。

`--constrained-head` 对应 LightNav 五位置动作语法，也是下表性能的测试条件。若应用使用
完整词表解码，必须单独验证输出和性能。

## 性能

测试条件：

- 客户提供的 35 帧有序图片；
- 每次请求为 9 帧历史加 1 帧当前画面，共 347 个 prompt token；
- 视觉缓存稳态为 4/1，关闭 tubelet 预取和上一请求 token 候选；
- Text 252 个投影与 Vision Encoder 均为 W8A16，使用五位置语法 head；
- 取 `frame_index >= 10` 的 25 次稳态请求中位数；
- 模型加载、图片初始处理和滑窗填充不计入稳态结果。

| 运行 | 准备 | Vision | Prefill | Decode 5 | Total | 帧间总耗时 | 输出一致性 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 完整 FP16 | 17.736 ms | 75.981 ms | 139.914 ms | 210.631 ms | **444.629 ms** | **447.406 ms** | 参考 |
| 激活感知 Text W8 + Vision W8，冷启动 1 | 16.757 ms | 29.608 ms | 91.199 ms | 115.749 ms | **254.346 ms** | **255.810 ms** | 35/35 帧，175/175 token |
| 激活感知 Text W8 + Vision W8，冷启动 2 | 16.671 ms | 30.099 ms | 91.299 ms | 115.850 ms | **254.227 ms** | **256.213 ms** | 35/35 帧，175/175 token |

各阶段分别取中位数，因此阶段中位数之和可能与 `Total` 有小量差异。

完整汇总见
[`benchmark_results/lightnav_activation_w8_20261118/summary.json`](../benchmark_results/lightnav_activation_w8_20261118/summary.json)。

## 正确性与限制

运行脚本会比较缓存路径与重算路径的 pooler/DeepStack 特征，FP16 最大绝对误差容限为
`1e-3`。如果出现差异超限，不要放宽容限，应先检查 RhinoForge commit、
`rpu_backend.so`、runtime 算子资产和模型 checkpoint 是否匹配。

35 帧、175 token 一致只证明当前离线样本上的逐 token 一致性。校准和输出边界补偿均
来自这组数据，不能替代独立数据集和闭环导航成功率评测。
