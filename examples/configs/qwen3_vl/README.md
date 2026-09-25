# Qwen3-VL 配置

本目录只有一套 `[example/input/run]` 配置，共 44 份：`text/` 20、`vision/` 3、`vl/` 21。

| 路径 | 范围 |
|---|---|
| `text/{2b,4b,8b}` | 纯文本 FP16、runtime W8/W4、显式 P192/D192 与 P4096/D4096 |
| `text/32b` | 精确 runtime W8/W4 |
| `vision/{2b,4b,8b}/fp16.toml` | 独立 Vision 组件 |
| `vl/{2b,4b,8b}` | 图文 FP16、runtime W8/W4、显式长度配置 |
| `vl/32b` | 精确 runtime W8/W4；`w8a16_legacy.toml` 单独保留兼容格式与 opt-in |
| `*/2b/w8a16_text.toml`、`*/{2b,4b}/w4a16_text_awq.toml` | Text-only 量化范围 |

## 运行与输入

准备完整 checkpoint、processor/tokenizer 和匹配的 [运行时资产](../../../docs/runtime_assets.md)，从仓库根目录执行：

```sh
python examples/run_model.py --config examples/configs/qwen3_vl/vl/4b/w8a16.toml --check-config
python examples/run_model.py --config examples/configs/qwen3_vl/vl/4b/w8a16.toml
```

`example.registry_alias` 在本地模型缓存中解析；可在本地配置设置 `input.checkpoint` 指向完整转换目录。路径相对于启动目录，`--check-config` 不加载权重或运行板卡。`example.profile_id` 绑定 [公开配置约束](../profiles.json)，不能只改 dtype 或 alias 来跳过格式检查。

`input.component` 选择 `text`、`vision` 或 `multimodal`。图文的 `input.images` 为有序图片路径列表，processor 生成缩放、归一化、patch、占位符和 grid；多图属于同一个请求，不是请求 batch。`input.prompt` 经模型 chat template 处理，不应手写或复制视觉 token 来凑长度。

`input.prefill_tokens` 包含视觉占位符，不能截断 processor 输入或任意裁剪 patch/grid。`decode_steps=D` 表示首 token 后的 D 次 decode，最多得到 D+1 个 token；`stop_on_eos`、`run.warmup_decode_steps` 和正式计时选项按选定文件保留。`run.warmup/runs/seed` 控制预热、样本数和种子。固定 P/D 配置不是自动运行的测试矩阵。

`assets/example.ppm` 是公开示意图片，不是历史测量输入。模板不覆盖视频、MoE 或请求 batch 大于一。分辨率与 grid 由 processor 和模型 envelope 共同约束，不能从 HF 最大上下文或任意图像尺寸推导运行能力。

## 量化范围

| 配置 | Text 投影与 head | Vision |
|---|---|---|
| `fp16.toml` | 原始浮点权重安装为 FP16 | FP16 |
| `w8a16.toml` | 文本 Q/K/V/O、gate/up/down 与独立 LM head 为逐输出通道 INT8；embedding 和 activation 为 FP16 | checkpoint 保持浮点，安装为 W8 |
| `w4a16.toml` | runtime W4：七投影按精确源格式转换，head 单独 RTN W4/G32 | 安装为 W8 |
| `w8a16_text.toml` | 仅 Text 七投影 W8，head 浮点 | 浮点 |
| `w4a16_text_awq.toml` | 源 AWQ Text 七投影，head 浮点 | 浮点 |

Runtime W4 不等于 Text-only AWQ，也不是全模型 W4。合规 AWQ 源保留七投影的整数与 scale，head 使用独立 RTN W4；来源和量化声明必须匹配加载器。2B/4B 的精确浮点源还可转换 RTN W4，这不含 AWQ 校准结果。8B runtime W4 要求匹配的 `compressed_tensors_awq` 来源，不能拿其他格式代替。

```sh
python -m rpu_backend.quant.convert_qwen3_vl \
  --bits 8 --src /path/to/Qwen3-VL-2B-Instruct --dst /path/to/qwen3-vl-2b-runtime-w8
python -m rpu_backend.quant.convert_qwen3_vl \
  --bits 4 --src /path/to/Qwen3-VL-2B-AWQ --dst /path/to/qwen3-vl-2b-awq-runtime-w4
```

源与目标使用不同目录，目标必须尚不存在。AWQ 源需为加载器接受的 `compressed-tensors` / `pack-quantized` / `compressed` 对称静态整数 W4/G32，权重分片、索引和 packed/shape/scale 齐全；不附带激活/KV 量化、稀疏或额外变换。8B 使用对应尺寸的源、目标与 [runtime W4 配置](vl/8b/w4a16.toml)。详见 [量化说明](../../../docs/quantization.md)。

32B 的 [兼容 W8 配置](vl/32b/w8a16_legacy.toml) 与 [runtime W8 配置](vl/32b/w8a16.toml) 是不同 checkpoint 合同。兼容格式仅量化 Text7，head、embedding 和 Vision 保持 FP16，仍要求 `example.opt_in.QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED="1"`。
该精确 TP8 格式使用独立的受控规划边界：prefill 物理执行长度上限 336、chunk 上限 64；336 不是 KV 容量或已验证的完整序列范围。逐层权重 bank 和冷态 Graph arena 准备保持原精度与所有权检查，不能借 runtime W8/W4 的 envelope 或权重替代该格式。
公开单图配置已完成一次 P78、D4、W1/R2 的真实 CLI、同 teacher 链 CPU 数值对照及 Graph 生命周期检查。这个有界结果不构成完整长度、任务质量或性能认证；Source-only / controlled / uncertified 状态和显式 opt-in 保持不变。

实际执行在 processor 和权重加载前绑定 checkpoint 元数据的尺寸与精度范围。
FP16 配置不能静默加载量化模型，legacy32 配置也不能借 runtime W8 权重绕过
自身的执行边界；兼容 checkpoint 的目录名称不受限制。

## 调度、精度与缓存

| 字段 | 含义 |
|---|---|
| `rpu_execution.model.num_cores` | 冷态拓扑，必须匹配具体模型和精度 profile。 |
| `prefill.linear_acc32` / `vision.linear_acc32` | 默认 ACC16，`true` 选择 ACC32；prefill 精度同时用于文本 decode。 |
| `prefill.fast_replay` | 2B/4B/8B 的文本和图文 FP16 模板显式设为 `true`，与性能脚本一致；API 省略此字段仍默认关闭。仅在原生 owner/Graph 检查满足时复用 prefill，没有独立 `vision.fast_replay`。 |
| `prefill.chunk_size` | 默认 `"auto"`；显式值须满足该模型 16 倍数及容量检查。 |
| `prefill.padding_budget` | planner 可搜索的额外 padding 预算，默认 64；不改变真实 token。 |
| `prefill.padding_rows` | `"auto"` 或精确非负整数；整数与 budget 互斥。 |
| `vision.chunk_size` | 默认 `"auto"`，不能绕过图像分组与单图容量。 |

默认 Vision grid 高、宽各不超过 48，单图 patch 还受默认 2048 行 Vision cache 限制，必须同时满足。2B/4B/8B 文本 prefill 物理执行上限为 4176 行，chunk 上限分别为 320/256/128；物理长度包含视觉 token 与 padding，最终以 planner、缓存和算子检查为准。

`fast_replay` 是加载时确定的选项，不能对已安装的模型热切换。Vision-only 模板不设置此文本选项；启用它不改变权重精度、输入长度限制或 profile 支持状态。

一个进程保留一个 RPU owner。复用模型和合法的 Graph 签名，正确重置请求的 KV 状态；不要逐 token 重建模型或清 Graph。改变冷态精度、拓扑和融合配置后重新加载。`run.profile` 会增加诊断开销，正常测量保持关闭；公开示例不采集硬件 trace。CPU 线程、缓存目录和 Launch 缓冲在启动进程前设置，详见 [运行时配置](../../../docs/runtime_config.md)。
