# 量化

简体中文 | [English](quantization.md)

RhinoForge 为部分模型家族提供离线 checkpoint converter。量化支持按配置判定：转换后
checkpoint 不继承其 FP16 来源或其他尺寸的状态。使用前检查[示例](model_support.zh.md)
和 release 的[模型资产](model_assets.zh.md)。

所有 converter 读取 source checkpoint 并创建新 destination；已有 destination 会拒绝，
失败时清理临时目录。保持原 checkpoint 不变，并记录源文件和转换文件的 hash。

## Qwen3 W8A16

Qwen3 converter 把七个 decoder projection weight 转成 signed INT8，并为每个 output
channel 保存一个 FP16 scale；其他浮点 tensor 默认保留 FP16。
14B 本地转换 recipe 使用 untied INT8 `lm_head` 和 FP16 embedding：

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

使用相同选项运行 verifier：

```bash
python -m rpu_backend.quant.verify_qwen3_w8a16 \
  --src /path/to/Qwen3-14B \
  --dst /path/to/qwen3-14b-w8a16-lmhead-int8 \
  --quant-lm-head
```

Verifier 检查 method、INT8/scale tensor 配对和代表性 dequantized weight 样本。
这些检查只覆盖转换一致性，不代表模型或设备准入。

## Qwen3 显式安装 W4/G32

公共 CausalLM loader 的 `quantization="w4a16"` 对精确 dense 0.6B、1.7B、4B、
8B、14B、32B 的七类 decoder 投影做一次安装期 W4/G32 转换；
`quantization="w4a16_lm_head"` 还转换 head。输入必须是原始浮点 checkpoint，
不是 AWQ/GPTQ 或离线 W8 产物。embedding、norm 和 activation 保持 FP16。
此入口固定 TP8、batch=1；`prefill.linear_acc32=false/true` 分别选择 ACC16/ACC32。
普通 FP16 入口的准入不会因此扩大，模板和转换也不等同于质量认证。
参见 [Qwen3 配置](../examples/configs/qwen3/README.md)。

## Qwen3-VL

2B/4B/8B 和精确 LightNav-0 配置的离线转换入口为
`python -m rpu_backend.quant.convert_qwen3_vl`。LightNav-0 的 `--bits 8` 只量化
Text7 投影，动作输出头和 Vision 保持 FP16；其他公开模型按原有 W8A16 配方转换。
实验参数 `--lightnav-vision-w8` 会同时按通道量化 Vision；它会改变模型输出，只有
完成真实导航准确率验证后才能用于部署。
使用 `awq_w4.toml` 时，
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

选择匹配的 [Pi0.5 配置](../examples/configs/pi05/README.md)，并随 checkpoint
保留生成的量化元数据。

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

# Runtime mixed W4A16-KV8 评估格式（不是纯 W4）。
python -m rpu_backend.quant.convert_pi05 \
  --src /path/to/pi05-source \
  --dst /path/to/pi05-mixed-W4A16-KV8 \
  --fake-w4 --real-w4
```

该格式对 q/o/gate/up/down 使用 per-channel signed INT4，K/V projection
保持 W8。权重以 INT8 tensor 保存，FP16 scale shape 为 `[N]`，加载时转换为
运行库要求的 packed 布局。Activation、KV cache、SigLIP、AdaRMS dense 和
Action projection 保持 FP16，processor sidecar 不变。

`--keep-int8` 接受逗号分隔 projection name，用于受控 W4 mixed-precision 实验。
`--real-w4` 是该 mixed profile 的遗留 CLI 名称，会自动让 K/V projection 权重
保持 W8。fake-W4 诊断张量与 runtime mixed-W4 格式分开使用。Converter 会删除
旧 remapped checkpoint，让 loader 从新量化 tensor 重建。

### Action NVFP4 优化格式

```sh
python -m rpu_backend.quant.convert_pi05 --src SOURCE_DIR --dst NVFP4_DIR --action-w4 --action-w4-format nvfp4
```

`--action-w4` 默认使用 NVFP4 `striped_v2` ABI、block16；Action Q/O/Gate/Up/Down
使用 NVFP4，K/V 和 VLM 保持 W8。元数据写入 `rpu_quant_config.json`。
选择两相机或三相机的
[`w8a16_action_nvfp4` 模板](../examples/configs/pi05/README.md)。
`w8a16_prefill_w8a8_action_nvfp4` 模板在安装时另选 Prefill GateUp A8。
Checkpoint 配方须与模板配套；`--real-w4` 生成上文所述的逐通道 INT4/KV8 格式。

Action 专用转换还接受以下显式格式：

| `--action-w4-format` | Action 权重与 scale 存储 |
|---|---|
| `nvfp4`（默认） | E2M1 编码、K16 FP8 scale 和独立 FP32 tensor scale；`striped_v2` ABI |
| `int4_g128` | INT8 `[N,K]` 保存 signed INT4，每个 K128 分组有独立 FP16 scale `[K/128,N]` |
| `legacy_int4` | 逐输出通道 signed INT4；打包时重复通道 scale |

从原始浮点权重转换到新目录时，同时传 `--action-w4` 和所需格式。
三者均写入 `int4_components=["expert"]`，Action K/V 和全部 VLM projection 保持 W8。
格式之间需要重新转换，不能只重命名目录或修改 metadata。

当前优化模板在安装前固定相机数、文本桶和累加模式。`input.text_tokens` 只选择物理桶，
不会分词、截断或填充输入。提供匹配的 `[1,T]` 整数 token、非空且只在右侧 padding 的 bool
mask，再用该 batch 准备 Graph。两相机输入需要两路真实图像；改变图数或文本桶应重建 policy。
FP16 模板要求浮点权重且无量化 metadata，并关闭 Vision、AdaRMS 动态量化。

## Wall-OSS

W8A16 converter 量化所有 floating 2-D weight，`--skip-modules` 匹配项除外；embedding
默认跳过：

```bash
python -m rpu_backend.quant.convert_wall_oss_w8a16 \
  --src /path/to/wall-oss-source \
  --dst /path/to/wall-oss-W8A16
```

W4 默认使用 `--group-size 32` 的 group-wise scale；`--group-size 0` 选择旧版
per-channel scale。正分组大小必须整除每个转换层的 input dimension：

```bash
# Per-channel W4。
python -m rpu_backend.quant.convert_wall_oss \
  --src /path/to/wall-oss-source \
  --dst /path/to/wall-oss-W4A16 \
  --group-size 0

# Group-wise W4，默认配置。
python -m rpu_backend.quant.convert_wall_oss \
  --src /path/to/wall-oss-source \
  --dst /path/to/wall-oss-W4A16-g32 \
  --group-size 32
```

当前 Wall-OSS W4 converter 拒绝非空 `--keep-int8`，因为 runtime 不消费该 mixed
precision metadata。两个 decoder expert 的投影均使用 INT4，包括 expert0 Down。

W4A16 runtime 只接受 packed group representation。输入宽度为 `K`、group size 为 32
时，现有 helper 会把一维 per-output-channel scale 重复成 `K / 32` 组，再生成
controller-striped layout；不要添加模型专用 packing 或 tile override。FP16/W8/W4
Linear 都使用共享 generated auto-tiling，见
[Linear 自动分块](../knowledge/concepts/linear-autotiling.md)。

## Qwen3.5 Dense

离线入口为 `python -m rpu_backend.quant.convert_qwen3_5 --src SOURCE_DIR --dst OUTPUT_DIR`。
统一文本投影使用 `--method w8a16` 或 `--method w4a16_pgrp`，混合配置使用 `--profile`：

| Profile | 文本投影 |
|---|---|
| `w4_main_w8_gdn` | 主投影 W4/G32；GDN QKV/Z/out W8；GDN B/A FP16 |
| `w8_main_w8_gdn` | 主投影和 GDN QKV/Z/out W8；GDN B/A FP16 |
| `w4_mlp_only` | MLP Gate/Up/Down W4/G32；Attention、GDN FP16 |
| `w4_mlp_w8_attn_gdn` | MLP Gate/Up/Down W4/G32；Attention、GDN W8 |

混合配置在 `config.json` 写入 `quant_config.format_version=2`。后两项覆盖每层 MLP，
固定 `--group-size 32`。Vision、embedding、LM head、norm 和 GDN convolution 保持 FP16。
源浮点张量直接转 FP16；旧转换产物需要从原始 checkpoint 重新生成，以消除此前的 BF16 中间转换。

```sh
python -m rpu_backend.quant.convert_qwen3_5 --src SOURCE_DIR --dst MIXED_DIR --profile w4_mlp_w8_attn_gdn
```

使用 `RPUModelForConditionalGeneration` 加载，以保留整数权重和 scale。
选择匹配的[文本或图像模板](../examples/configs/qwen3_5/README.md)，runner 会在权重加载前
检查量化元数据。量化图像模板要求同时设置 `QWEN3_5_QUANT_ALLOW_UNCERTIFIED=1` 和
`QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1`。

独立的 QuantTrio AWQ G128 来源保持第零层 MLP、Attention、GDN、Vision、LM head 为 FP16，
后续层 MLP 保留 AWQ 权重。用对应 AWQ 模板加载原始 checkpoint；上述 RTN converter 不生成此格式。

## Qwen3.5 MoE mixed E4M3

使用独立 CPU 转换器从官方未量化 `Qwen/Qwen3.5-35B-A3B` 生成权重。
从仓库根目录执行，指定已有浮点源和尚不存在的输出目录：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5_moe_fp8.py \
  --src SOURCE_DIR --dst OUTPUT_DIR --expert-chunk 1 --threads 8
```

[完整下载与运行命令](../examples/configs/qwen3_5/README.md#cmd-moe-mixed)
默认输出到 `$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-rpu-fp8-mixed`。转换需要 PyTorch、
safetensors 和支持 `Qwen3_5MoeConfig` 的 Transformers，无需校准数据或设备运行时。

脚本量化 210 个 routed-expert、full-attention Q/K/V/O 和 GDN QKV/Z/out 张量，
保存 E4M3 原始 UINT8 codes 与逐输出通道 FP16 scales；其余浮点张量，包括 router、
shared expert、head、embedding、norm 和 Vision，保存为 FP16。输出保留 config、
tokenizer、processor 和分片元数据。官方 block-FP8 `weight_scale_inv`、GPTQ 和 RTN
包不能作为输入。Text/VL、核数及 ACC 模板共用这份产物，各自准入规则保持不变。
新转换产物不自动继承历史派生资产的身份或测量结果。

## Qwen3.5 MoE routed INT4

35B-A3B 的 RTN 和官方 GPTQ 配置只量化 routed Gate/Up/Down expert，格式为 UINT4/G128。
Attention、GDN、router、shared expert、embedding、norm 和 LM head 使用 FP16。
两者均要求 batch=1、八核、Text ACC16；拒绝 `prefill.linear_acc32=true`。
Vision 的累加选项独立设置。

| 来源 | Checkpoint 要求 |
|---|---|
| RTN | 精确 `rtn_metadata`、indexed safetensors、完整 `derivation_manifest.json`；不使用校准 |
| 官方 GPTQ | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`，对称四位 G128、无 act-order、仅 routed expert 量化 |

RTN 的 zero-minus-one 为 `0x77777777`，官方 GPTQ 的 zero8 为 `0x88888888`。
两个 loader 分别校验其约定，再进入共享 UINT4 路径；GPTQ 保留原始整数 codes 和 FP16 scales。
存在但不合法的 `rtn_metadata` 不会回退到 GPTQ。这两个入口不接受任意 GPTQ 格式或 mixed-E4M3。

RTN 先将源分组提升为 FP32，计算 `s=max(abs(group))*(2/15)`，全零组使用 `2/15`；
随后 `q=clamp(round_even(w/s)+8,0,15)`。先确定编码，再把 scale 转成 FP16，保留下溢后的零。
反量化为 `(q-8)*s`。源文件使用 INT32 `qweight[K/8,N]`、FP16 `scales[K/128,N]` 和
`g_idx[k]=k//128`，RTN/GPTQ 的零点约定各自校验。TP8 下 Gate/Up 保持 G128；Down 将
每个原 G128 scale 重复成两个物理 G64 分组，编码与反量化结果不变。安装时按配套的两份算子库打包。

文本别名为 `RPU_MODEL_CACHE` 下的 `qwen3_5-35b-a3b-rtn-w4a16` 和
`qwen3_5-35b-a3b-gptq-w4a16`，也可用 `--model` 指定 checkpoint。
GPTQ 读取原始 indexed checkpoint，不接受 RTN derivation metadata；RTN 拒绝未完成的 staging
目录。`input_ids` 为 CPU `int64 [1,T]`；安装后通过 `Qwen3_5MoeCache.from_model` 为完整请求分配 cache。

使用条件生成 loader 和 `Qwen3_5MoeCache`。安装时逐层读取专家权重，不把全部专家展开为 FP16。
indexed reader 校验 shard 路径、header、shape、dtype 和 tensor 清单；完整文件 hash 在资产准备时
验证。运行时 checkpoint identity 记录元数据与文件状态，不代替完整 payload 校验和。

```sh
python -m examples.run --config examples/configs/qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128.toml --model RTN_DIR
python -m examples.run --config examples/configs/qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128.toml --model GPTQ_DIR
```

对应 `vl/35b_a3b/` 模板支持图像：在 cache position=0 同时传入 `input_ids`、`pixel_values`、
`image_grid_thw`、`mm_token_type_ids`，且不超过已安装 Vision KV 和 RoPE 边界。不支持视频。
模型安装一次，请求间调用 `cache.reset()`；prefill 使用有界 one-shot，decode 复用 Graph。
FP16 计算后的 logits 以 CPU FP32 返回。

## RhinoVLA v3 expert

[v3 模板](../examples/configs/rhinovla/v3/README.md) 默认使用内置
`rpu_backend.adapters.rhinovla.factory:v3_runtime_factory`，读取固定 v3 源码、checkpoint
和预处理 CPU 输入。`w8a16_action.toml` 量化 action Q/K/V/O、Gate/Up/Down；
`w8a16_action_adarms.toml` 还量化 AdaRMS condition 投影。两者均使用 INT8 权重、FP16 scale
和 activation，Vision、prefix language、action IO 保持所选浮点格式。自定义可信 factory
需满足相同输入与生命周期合同；runner 检查实际 expert 存储。

`w8a16_pipeline.toml` 进一步量化 Vision、Text、action IO 和最终 AdaRMS 条件投影；
HIGH/vector QK 档保持同一权重配方，调整非线性与归一化执行。这四档 W8 与 FP16
均有对应 `rtc_*.toml`：RTC 保留量化范围，使用所选投影权重和 scale 准备各 delay
的条件参数。使用内置 v3 factory，传入动作前缀与 `delay=0..6`；输入边界及验证状态见
[RTC 说明](../examples/configs/rhinovla/v3/README.md#rtc)。RTC 无需另外生成量化权重目录。

## 运行时转换 helper

Hy-Embodied W8/W4 在运行时转换，没有离线导出 CLI。输入和转换配方见
[Hy-Embodied 指南](../examples/configs/hy_embodied/README.md)。

`rpu_backend.quant.convert_qwen3_w4a16` 是 Qwen3 0.6B 的 in-memory 评估 helper，
不是通用离线 `--src/--dst` converter，也不生成可分发 checkpoint；除非精确 release
配置另行声明，应视为 Source-only。

公开 `rpu_backend.quant` 导出 `quantize_linear_per_channel` 和
`dequantize_linear_per_channel` 供 converter 作者使用。INT4 packing helper 属于实现层；模型范围见[示例](model_support.zh.md)。

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
6. 设备对精确量化同 dtype 参考验证实现一致性，再按
   [验证策略](validation_policy.zh.md)对固定 FP16/CPU FP32 anchor 检查任务质量；
7. 验证 warmup 和 Graph lifecycle；
8. 在新进程运行公开 E2E TOML example。

量化结果应与浮点来源分别记录，并注明精确 checkpoint 和执行配置。
E2E 准入使用对应[模型配置](model_support.zh.md)及其验证要求。
