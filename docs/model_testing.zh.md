# 模型执行与性能采集

简体中文 | [English](model_testing.md)

RhinoForge 提供两层手工执行入口：

- `examples/` 下的脚本是最短、最易读的模型入口；
- `examples/run_model.py` 调用同一批脚本，并从 TOML 应用已记录的运行时环境以及可选的 Torch、RPU 硬件 profile。

这些入口用于板卡冒烟和诊断。一次成功运行只能证明所选本地配置能够执行；它不能替代 CPU/golden 数值对比，也不会扩大[模型支持范围](model_support.zh.md)中的状态。

## 入口清单

| 模型/配置 | 直接入口 | 统一 runner 目标 | 可用范围 |
|---|---|---|---|
| Qwen3 FP16、Llama-3.2-1B | `examples/causal_lm.py` | `causal_lm` | 有匹配的精确发布权重和配置时可运行 |
| Qwen3 14B W8A16 | `examples/causal_lm.py` | `causal_lm` | Source-only 本地转换入口；v1.0.0 未绑定公开不可变派生 checkpoint 身份或 hash |
| Qwen3.5 文本 | `examples/qwen3_5_text.py` | `qwen3_5_text` | 可运行文本入口 |
| Qwen3.5 Vision 2B/4B | `examples/qwen3_5_vision.py` | `qwen3_5_vision` | Experimental/numeric-blocked 图像入口；受控评估必须精确设置 `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1`，不构成支持声明 |
| Qwen3-VL 2B/4B | `examples/qwen3_vl.py` | `qwen3_vl` | 可运行图文入口 |
| Qwen3-VL 8B dense FP16 | `examples/qwen3_vl.py` | `qwen3_vl` | Limited 精确 padded profile 入口；4/4 numeric/task/Graph 已通过，包含 dual-full，base 5/5、long-text 9/9 token 精确一致；相邻长度、视频和量化路径不继承 |
| Qwen3-VL 32B W8A16 | `examples/qwen3_vl.py` | `qwen3_vl` | 仅源码诊断入口；显式 graph-blocked opt-in 只用于复现第三个 decode step 起的已知全零 logits |
| DINOv3 ViT-B | `examples/dinov3.py` | `dinov3` | 可运行视觉入口 |
| SigLIP | `examples/siglip.py` | `siglip` | 可运行组件入口，使用匹配 Pi0.5 包中的视觉塔 |
| Pi0.5 Libero FP16 | `examples/pi05.py` | `pi05` | Supported 精确 profile；9/9 numeric/task 与 27/27 execution-plan record 已通过，另有独立 fresh-process Graph 生命周期门通过 |
| Pi0.5 Libero W8A16 | `examples/pi05.py` | `pi05` | Source-only 本地转换入口；未绑定公开不可变派生 checkpoint 身份或 hash |
| Pi0.5 Libero W4A16 | `examples/pi05.py` | `pi05` | Source-only 本地转换入口；未绑定公开不可变派生 checkpoint 身份或 hash |
| Pi0.5 减步数/闭环 | 公开策略 API | 无 | Source-only 且验证 pending；不提供公开冒烟声明，也不继承精确 FP16 或量化路径的结论 |
| Pi0.5 base | 无 | 无 | 仅源码别名；没有独立任务配置或冒烟声明 |
| Wall-OSS FP16/W8A16/W4A16 | `examples/wall_oss.py` | `wall_oss` | 公开 `wall-x` source-only 接入 |
| Hy-Embodied-0.5-VLA FP16/W16 | `examples/hy_embodied.py` | `hy_embodied` | 精确公开 profile 的可运行归一化 action 入口 |
| Hy-Embodied-0.5-VLA W8/W4 | `examples/hy_embodied.py` | `hy_embodied` | Source-only runtime 转换入口；未绑定公开不可变派生 checkpoint 身份或 hash |
| RhinoVLA | `examples/rhinovla.py` | `rhinovla` | 集成入口；模型仓库提供运行时工厂 |
| Gemma4-E4B 文本 | `examples/gemma4.py` | `gemma4` | 仅源码文本入口；需要兼容的本地权重和依赖集合 |
| GR00T-N1.7-3B | `examples/gr00t.py` | `gr00t` | 仅源码 builder 入口，输入为调用者预处理的张量 |
| LingBot-VLA-V2 | `examples/lingbot2.py` | `lingbot2` | 公开 source-only 接入，图像由调用方提供 |
| Galaxea G0.5 | `examples/g05.py` | `g05` | 配置和集成边界检查；模型仓库提供正式 CPU policy 对象前执行保持 fail-closed |
| InternVLA-N1 + NavDP | `examples/internvla_navdp.py` | `internvla_navdp` | 调用者固定 manifest 和 embedding 的受控 NavDP 组件；不是组合端到端入口 |
| Qwen3 0.6B/1.7B/4B/8B W8A16、Qwen3 32B | `model_support.zh.md` 指定的公开 API | 无 | 仅源码；没有已发布的可运行配置或冒烟配置 |
| 不支持的配置 | 无 | 无 | 不可运行 |

表格只记录真实的公开入口；仅存在源码不会被列为测试路径。

## 仓库内配置模板

下表每个 stem 在 [`examples/configs`](../examples/configs) 下都只有一份 `.toml`。同一文件包含直接模型请求、runner 环境、Torch/硬件 profiling，以及目标接受的全部执行阶段。配置内容不会改变其支持状态。

| 系列 | 配置 stem |
|---|---|
| Qwen3 | `qwen3_0_6b`、`qwen3_1_7b`、`qwen3_4b`、`qwen3_8b`、`qwen3_14b_w8a16` |
| Llama | `llama_3_2_1b` |
| Qwen3.5 文本 | `qwen3_5_0_8b`、`qwen3_5_2b`、`qwen3_5_4b`、`qwen3_5_9b` |
| Qwen3.5 Vision | `qwen3_5_vision_2b`、`qwen3_5_vision_4b` |
| Qwen3-VL | `qwen3_vl_2b`、`qwen3_vl_4b`、`qwen3_vl_8b`、`qwen3_vl_32b_w8a16` |
| DINOv3 / SigLIP | `dinov3_vit_b`、`siglip` |
| Pi0.5 Libero | `pi05_libero`、`pi05_libero_w8a16`、`pi05_libero_w4` |
| Wall-OSS | `wall_oss`、`wall_oss_w8a16`、`wall_oss_w4a16` |
| Hy-Embodied / RhinoVLA | `hy_embodied`、`rhinovla` |
| Gemma4 / GR00T | `gemma4`、`gr00t` |
| LingBot2 / G0.5 | `lingbot2`、`g05` |
| InternVLA-N1 + NavDP | `internvla_navdp` |

量化模板只用于本地 converter/runtime 路径；模板存在不会绑定派生
checkpoint 身份，也不会改变 Source-only 状态。

## 直接示例

所有直接示例都接受 TOML，并可在不加载权重的情况下验证配置：

```bash
python examples/qwen3_vl.py --config examples/configs/qwen3_vl_2b.toml --check-config
python examples/qwen3_vl.py --config qwen3_vl.local.toml
```

复制 `examples/configs/` 下最接近的文件，再设置本地权重和输入路径。权重获取与哈希方式见[模型资产](model_assets.zh.md)。

通用视觉和 VLM 模板使用 `assets/example.ppm`；这是仓库内的极小合成图，只用于让公开入口不依赖私有图片即可运行。数值或任务验证前必须替换为代表性图像。VLA 相机输入仍由调用方提供，因为相机名和预处理属于精确模型 profile。

Pi0.5 的 `batch_file` 不随仓库分发。请使用与权重匹配的 LeRobot policy 预处理流水线生成它，再用 `torch.save` 保存张量字典。该字典至少包含权重 `image_features` 配置中每个 key 对应的一份 batched image tensor，以及 `observation.language.tokens` 和 `observation.language.attention_mask`；预处理流水线也可保留该配置拥有的其他字段。相机名称、分辨率和 token 长度属于具体配置，应从权重/配置读取，不要照搬其他配置的 shape。

GR00T 入口同样需要调用者生成的张量映射。`input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw` 和 `state` 必须来自与权重匹配的预处理流水线。NavDP 组件入口需要调用者固定的“路径到 SHA256”JSON manifest，以及包含 `goal_embed [1,1,384]` 和 `rgbd_embed [1,32,384]` 的张量映射。这些都是输入，不是 RhinoForge 分发的模型资产。

Qwen3.5 Vision 默认拒绝执行：公开 example 会在加载模型权重前拒绝，共享 adapter 会在创建 Vision handle 或转换 Vision 权重前拒绝。仅在受控评估时，于新进程启动前显式设置：

```bash
QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1 \
  python examples/run_model.py --config examples/configs/qwen3_5_vision_2b.toml
```

该取值不豁免已失败的官方真实图数值门，也不启用视频。Qwen3-VL 32B 模板保留显式 opt-in，只用于复现已知 Graph 硬失败；它不代表硬门通过或可运行支持。LingBot2 和 InternVLA/NavDP 模板包含明确的 source-only 或受控评估确认项；删除门禁时必须失败，不能退化为普通配置。G0.5 没有稳定的独立公开 policy 构造器：其示例验证配置后，会明确给出正式模型仓库需要完成的集成步骤并停止。对这些入口执行成功的 `--check-config` 只验证配置结构，不会提升支持状态。

## 完整 TOML runner

每个可运行目标在 `examples/configs/` 下都有一份 TOML。同一文件包含模型/请求字段、公共运行时配置、Torch profile、硬件 profile，以及适用的全部 `rpu_execution` 阶段。完整字段和归属说明见[运行时配置](runtime_config.zh.md#toml-参数目录)。

以 `examples/configs/qwen3_0_6b.toml` 为例：

```bash
python examples/run_model.py --list-targets
python examples/run_model.py \
  --config examples/configs/qwen3_0_6b.toml \
  --check-config
python examples/run_model.py \
  --config examples/configs/qwen3_0_6b.toml
```

配置会把运行时环境、每 handle 执行规划、模型输入和 Torch profiling 合在一起：

```toml
[runner]
target = "causal_lm"

[runner.env]
RPU_LOG_LEVEL = 3
RPU_WARMUP = 1

[runner.torch_profile]
enabled = false
output = "profiles/torch.json"
record_shapes = false
profile_memory = false
with_stack = false
```

其余 `[model]`、`[generation]` 或 `[request]` 表由选中的直接示例读取。`[runner.env]` 只接受[运行时配置](runtime_config.zh.md)中记录的变量。TOML 布尔值会转换成可移植的环境值 `1`/`0`。runner 会在导入 `torch` 或 `rpu_backend` 前应用它们；每个进程只运行一个模型/配置。凭据、无关的进程加载器变量和 `RPU_KERNEL_LIB_PATH` 都不能写入 TOML；经批准的 Rhino Launch 库和合并算子资产应在部署环境中设置。“运行时配置”页面的**仅诊断变量清单**同样不能写入 runner；开发者必须在 shell 中逐项显式启用，并在分享前检查产生的本地文件。

每 handle 的执行规划放在顶层表中：

```toml
[rpu_execution.prefill]
chunk_size = "auto"
padding_budget = 64
```

统一 runner 验证该表并转交给现有模型入口。可接受的阶段如下：

| 目标 | 接受的 `rpu_execution` 阶段 |
|---|---|
| `causal_lm`、`qwen3_5_text`、`qwen3_5_vision` | `prefill` |
| `gemma4` | `prefill`（仅 `chunk_size`） |
| `qwen3_vl` | `prefill`、`vision` |
| `dinov3`、`siglip` | `vision` |
| `pi05`、`wall_oss`、`gr00t` | `prefill`、`vision`、`action` |
| `rhinovla` | 由可信运行时工厂声明，并通过能力握手检查 |
| `hy_embodied`、`lingbot2`、`g05`、`internvla_navdp` | 无；非空表会被拒绝 |

除 Gemma4 的窄范围外，`prefill` 接受 `chunk_size`、`padding_rows` 和 `padding_budget`；`vision` 与 `action` 接受 `chunk_size`。公开 loader 始终是配置边界的权威检查点，会在安装到 RPU 前拒绝不支持的值。

## Torch profile

设置 `[runner.torch_profile].enabled = true`，可为整个命令导出一份 Chrome trace。若安装的 PyTorch 同时暴露 CPU 与 `PrivateUse1` activity，runner 会请求二者。trace 包含模型初始化和推理；排查设备路径时可筛选 `rpu::` 以及命名的图范围。

`record_shapes`、`profile_memory` 和 `with_stack` 默认均为 `false`。只在需要时单独开启：shape、分配事件、栈帧和源码路径会向 trace 增加应用细节。

可使用 Perfetto 或其他 Chrome trace 查看器打开 JSON。profiling 会增加开销，因此延迟测量应使用关闭 profiler 的运行。

## 调试错误或不稳定运行

应先调试正确性，再采集性能 trace。保留一份不可变运行记录，其中包括权重/配置哈希、TOML、输入张量、执行参数和依赖版本。对随机策略，应只采样一次初始噪声或其他随机状态，再把完全相同的张量克隆给 CPU 与 RPU 路径；连续两次 RNG 调用得到的是不同输入。

使用能够区分问题的最小公开检查：

1. 比较第一次调用和相同 signature 的重复调用。只有重复调用变化时，应优先检查 Graph 生命周期、持久 SPM 或 DMA 地址所有权。
2. 保留第一份输出，运行相同 shape 的另一输入，再确认已保留输出的字节和 storage 相互独立。
3. 修改 capture 路径前，检查 `GraphCache.snapshot()`、`cache_invariant_ok()`、`debug_bucket_counts()` 和 `explain_miss(signature)`。仅有干净日志不能证明已经 BUILD 后稳定 REPLAY。
4. 隔离第一个发生偏差的公开组件或算子；在调试完整模型前，先用相同 CPU 输入对比 shape、dtype、layout 和数值。
5. 只按需提高 `torch.rpu.set_debug_level(...)`。使用 `torch.rpu.spm_alloc_dump("label")` 查看分配器摘要；只在受控本地对比中导出调试张量。

调试导出可能包含权重、用户输入派生数据和激活。请将其保存在仓库之外，调查结束后清理，并分享脱敏摘要而非原始张量。正确性和 replay 稳定后，再启动新进程，只开启测量问题所需的 profiler。

## 发布质量的模型检查还需要什么

支持声明必须把所有结果绑定到同一源码提交、Rhino Launch 包、合并算子资产、权重修订版本、输入边界、TOML 和输出哈希。应与 CPU 或批准的 golden 结果对比，覆盖 prefill 与 decode 或完整 policy 调用，warmup 后重复执行，并验证适用的图生命周期。未执行的门禁应记录为待完成，不能从冒烟运行推断。

请使用[模型验证策略](validation_policy.zh.md)中相互独立的硬语义、同 dtype parity、FP32 锚点和代表性任务门禁。单个最差行 cosine 只是诊断证据，不是全仓统一的产品结论。
