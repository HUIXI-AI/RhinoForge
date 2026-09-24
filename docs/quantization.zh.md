# 量化

简体中文 | [English](quantization.md)

RhinoForge 为部分模型家族提供离线 checkpoint converter。量化支持按配置判定：转换后
checkpoint 不继承其 FP16 来源或其他尺寸的状态。使用前检查[示例](model_support.zh.md)
和 release 的[模型资产](model_assets.zh.md)。

所有 converter 读取 source checkpoint 并创建新 destination；已有 destination 会拒绝，
失败时清理临时目录。保持原 checkpoint 不变，并记录源文件和转换文件的 hash。

## Qwen3 W8A16

Qwen3 converter 把七个 decoder projection weight 转成 signed INT8，并为每个 output
channel 保存一个 FP16 scale；其他浮点 tensor 默认保留 FP16。v1.0.0 中所有
生成的 W8A16 checkpoint 均为 Source-only，因为未绑定公开不可变派生
checkpoint 身份或 hash。14B 本地转换 recipe 使用 untied INT8 `lm_head`和
FP16 embedding：

```bash
python -m rpu_backend.quant.convert_qwen3 \
  --src /path/to/Qwen3-14B \
  --dst /path/to/qwen3-14b-w8a16-lmhead-int8 \
  --quant-lm-head
```

选项：

- `--skip-modules NAME [NAME ...]` 排除匹配 decoder projection；
- `--quant-lm-head` 写入 untied quantized `lm_head`；
- `--quant-embed-tokens` 同时量化 embedding，并要求 `--quant-lm-head`；此路径仍为 Source-only。

所有 Qwen3 W8A16 converter 输出在 v1.0.0 中均保持 Source-only。

使用相同选项运行 verifier：

```bash
python -m rpu_backend.quant.verify_qwen3_w8a16 \
  --src /path/to/Qwen3-14B \
  --dst /path/to/qwen3-14b-w8a16-lmhead-int8 \
  --quant-lm-head
```

Verifier 检查 method、INT8/scale tensor 配对和代表性 dequantized weight 样本。默认最多
21 个样本、minimum weight cosine `0.999`；这些是 converter 检查，不是模型 acceptance。
用 `--max-samples`、`--min-cosine` 设置 release procedure 值，再运行端到端验证。

## Qwen3 显式安装 W4/G32

公共 CausalLM loader 的 `quantization="w4a16"` 对精确 dense 0.6B、1.7B、4B、
8B、14B、32B 的七类 decoder 投影做一次安装期 W4/G32 转换；
`quantization="w4a16_lm_head"` 还转换 head。输入必须是原始浮点 checkpoint，
不是 AWQ/GPTQ 或离线 W8 产物。embedding、norm 和 activation 保持 FP16。
此入口固定 TP8、batch=1；`prefill.linear_acc32=false/true` 分别选择 ACC16/ACC32。
普通 FP16 入口的准入不会因此扩大，模板和转换也不等同于质量认证。
参见 [Qwen3 配置](../examples/configs/qwen3/README.md)。

## Qwen3-VL

2B/4B/8B 的离线转换入口为 `python -m rpu_backend.quant.convert_qwen3_vl`。
`--bits 8` 从公开 FP16 权重生成 W8A16。使用 `awq_w4.toml` 时，
`--bits 4` 的输入应为 symmetric G32 compressed-tensors AWQ：保留 Text AWQ，
LM head 转为 RTN W4，Vision 为 W8。转换器也接受精确匹配的公开 2B/4B/8B FP16
配置配合 `--bits 4`，但生成的是 Text/LM head RTN W4，不是 AWQ；两种 recipe 的
元数据不同。输入格式不匹配时会提前拒绝。

```sh
python -m rpu_backend.quant.convert_qwen3_vl --bits 8 --src SOURCE_DIR --dst W8_DIR
python -m rpu_backend.quant.convert_qwen3_vl --bits 4 --src AWQ_DIR --dst W4_DIR
```

加载转换目录时不需要额外的精度环境变量。原始 AWQ 与转换输出目录是不同资产。

## Pi0.5

默认 Pi0.5 转换把 VLM decoder 和 action-expert decoder projection 转为 W8A16。Vision
encoder、AdaRMS dense、action projection 和 processor sidecar 保持 FP16 或不变。

Pi0.5 W8A16 和 mixed W4A16-G32-KV8 输出在 v1.0.0 中均为 Source-only：
发布版本未为它们绑定公开不可变派生 checkpoint 身份或 hash；精确同量化 oracle
与 checkpoint 自有任务证据也仍待完成。

```bash
python -m rpu_backend.quant.convert_pi05 \
  --src /path/to/pi05-source \
  --dst /path/to/pi05-W8A16
```

还提供两个 W4 评估格式：

```bash
# 用 INT8 tensor 保存 signed 4-bit，仅数值 probe。
python -m rpu_backend.quant.convert_pi05 \
  --src /path/to/pi05-source \
  --dst /path/to/pi05-fake-W4 \
  --fake-w4

# Runtime mixed W4A16-G32-KV8 评估格式（不是纯 W4）。
python -m rpu_backend.quant.convert_pi05 \
  --src /path/to/pi05-source \
  --dst /path/to/pi05-mixed-W4A16-G32-KV8 \
  --fake-w4 --real-w4
```

该 runtime profile 是 mixed W4A16-G32-KV8，并非纯 W4。仅量化声明的 Gemma
VLM/action-expert Linear projection 权重：q/o/gate/up/down 沿 K 使用 symmetric
W4 group-size-32，K/V projection 权重保持 W8；activation 与 KV cache 均保持
FP16。Checkpoint 中逻辑 FP16 scale shape 为 `[K/32, N]`，加载时再转成 packed
pgrp ABI 的 controller-striped 布局。SigLIP、AdaRMS dense、action projection
和 processor sidecar 保持 FP16 或不变。

`--keep-int8` 接受逗号分隔 projection name，用于受控 W4 mixed-precision 实验。
`--real-w4` 是该 mixed profile 的遗留 CLI 名称，会自动让 K/V projection 权重
保持 W8。两种 W4 都是 Source-only 评估路径；提升前需有独立不可变资产身份、
精确同量化 oracle 与 checkpoint 自有任务证据。Converter 会删除旧 remapped
checkpoint，让 loader 从新量化 tensor 重建。

### Action NVFP4 优化格式

```sh
python -m rpu_backend.quant.convert_pi05 --src SOURCE_DIR --dst NVFP4_DIR --action-w4 --action-w4-format nvfp4
```

`--action-w4` 默认使用 NVFP4 `striped_v2` ABI、block16；Action Q/O/Gate/Up/Down
使用 NVFP4，K/V 和 VLM 保持 W8。元数据写入 `rpu_quant_config.json`。
这是 `optimized_profile.precision=w8_action_nvfp4` 的输入格式；两相机或三相机的
`w8_prefill_a8_action_nvfp4` 在安装时另选 Prefill GateUp A8。
它与上述 `--real-w4` INT4 G32/KV8 格式不可互换。

## Wall-OSS

W8A16 converter 量化所有 floating 2-D weight，`--skip-modules` 匹配项除外；embedding
默认跳过：

```bash
python -m rpu_backend.quant.convert_wall_oss_w8a16 \
  --src /path/to/wall-oss-source \
  --dst /path/to/wall-oss-W8A16
```

W4 默认使用 per-channel scale；`--group-size` 为正时使用 group-wise scale，并且必须
整除每个转换层的 input dimension：

```bash
# Per-channel W4。
python -m rpu_backend.quant.convert_wall_oss \
  --src /path/to/wall-oss-source \
  --dst /path/to/wall-oss-W4A16

# Group-wise W4；32 是现有评估配置。
python -m rpu_backend.quant.convert_wall_oss \
  --src /path/to/wall-oss-source \
  --dst /path/to/wall-oss-W4A16-g32 \
  --group-size 32
```

当前 Wall-OSS W4 converter 拒绝非空 `--keep-int8`，因为 runtime 不消费该 mixed
precision metadata。Wall-OSS W4 保持 Source-only。

W4A16 runtime 只接受 packed group representation。输入宽度为 `K`、group size 为 32
时，现有 helper 会把一维 per-output-channel scale 重复成 `K / 32` 组，再生成
controller-striped layout；不要添加模型专用 packing 或 tile override。FP16/W8/W4
Linear 都使用共享 generated auto-tiling，见
[Linear 自动分块](../knowledge/concepts/linear-autotiling.md)。

## Source-only helper

Hy-Embodied W8/W4 是 runtime 派生的评估路径。v1.0.0 未为它们绑定公开
不可变派生 checkpoint 身份或 hash，因此状态为 Source-only；公开 FP16/W16
profile 的状态不转移。

`rpu_backend.quant.convert_qwen3_w4a16` 是 Qwen3 0.6B 的 in-memory 评估 helper，
不是通用离线 `--src/--dst` converter，也不生成可分发 checkpoint；除非精确 release
配置另行声明，应视为 Source-only。

公开 `rpu_backend.quant` 导出 `quantize_linear_per_channel` 和
`dequantize_linear_per_channel` 供 converter 作者使用。INT4 packing helper 属于实现层，
其存在不代表模型支持。

## Metadata 与加载

Qwen3 在 `config.json` 的 `quant_config` 记录 W8A16；Pi0.5、Wall-OSS 在 checkpoint
旁写 `rpu_quant_config.json`。Loader 按模型检查 method、tensor 和 profile。不要手改
metadata 或重命名 scale tensor。

量化不会加密模型权重，也不改变 license。遵守 source model 条款和
[模型资产](model_assets.zh.md)记录的分发策略。

## 验证清单

发布或选择量化配置前：

1. 固定并 hash 每个 source checkpoint 文件；
2. 在新 destination 运行 converter 并保存完整 log；
3. 检查输出 manifest、metadata、tensor name/dtype/shape 和 file hash；
4. 对比采样或完整 dequantized weight 与 FP16 来源；
5. 通过公开 RhinoForge 入口加载，并确认 mismatch profile fail fast；
6. RPU 对精确量化同 dtype 参考验证实现一致性，再按
   [验证策略](validation_policy.zh.md)对固定 FP16/CPU FP32 anchor 检查任务质量；
7. 验证 warmup 和 Graph lifecycle；
8. 在新进程运行公开 E2E TOML example。

量化结果应与浮点来源分别记录，并注明精确 checkpoint 和执行配置。
[示例](model_support.zh.md)存在或 weight cosine 通过均不等于 E2E 支持。
