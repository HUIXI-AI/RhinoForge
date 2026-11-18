# RhinoVLA v3 源码入口

<a id="rhinovla-v3-受控配置"></a>

[v3 模型页](../README.md) · [模型配置表](../../README.md#configurations) · [权重准备](../../../WEIGHTS.md)

本目录的 TOML 直接从 v3 `steps_150000` non-RTC 源码、checkpoint 和 CPU fixture 构建模型，用 `examples/run.py` 运行。每份文件对应一组固定的冷启动开关（`[model.opt_in]`），与登记的 example-only profile 完全一致，修改开关会被拒绝。20 个开关全部写出，关闭的写为 `"0"`；shell 中继承的同名环境变量若与之不同，运行时直接报错。精度同时显式传给 factory，构建后再检查已安装的 expert 权重，与 `scripts/validate_rhinovla_v3.py` 的检查一致。同一文件也可交给 `validate_v3.py --config` 做运行时校验。

| 配置类别 | 配方与入口 |
|---|---|
| 基础 | [FP16](fp16.toml)、[W8A16 Action](w8a16_action.toml) |
| Pipeline | [W8A16](w8a16_pipeline.toml) 及当前目录中的 input-cache、high、fused、vector Q/K、partial-RoPE 变体 |

Pipeline 变体按下列顺序逐个叠加开关，每一份比上一份多开表中的开关：

| 配置 | 新开的 `[model.opt_in]` 开关 | 动作输出 |
|---|---|---|
| [input-cache](w8a16_pipeline_input_cache.toml) | `RPU_RHINOVLA_PRECOMPUTE_GATE_TANH`、`RPU_RHINOVLA_VISION_INPUT_CACHE`，并在 `[rpu_execution.prefill]` 写 `padding_rows = 10` | 与 W8A16 不同 |
| [high](w8a16_pipeline_high.toml) | `RPU_RHINOVLA_HIGH_PRECISION` | 与 input-cache 不同 |
| [fused](w8a16_pipeline_high_fused.toml) | `RPU_RHINOVLA_HIGH_PRECISION_FUSIONS` | 与 high 逐位相同 |
| [vector K](w8a16_pipeline_high_fused_vector_k.toml) | `RPU_RHINOVLA_VECTOR_K_NORM` | 与 high 逐位相同 |
| [vector QK](w8a16_pipeline_high_fused_vector_qk.toml) | `RPU_RHINOVLA_VECTOR_Q_NORM` | 与 high 逐位相同 |
| [partial-RoPE](w8a16_pipeline_high_fused_vector_qk_partial_rope.toml) | `RPU_RHINOVLA_PARTIAL_ROPE` | 与 high 逐位相同 |

high 之后的 4 个开关只改变算子融合、Q/K 归一化的向量化方式和 RoPE 的实现，保留 HIGH 的计算和 FP16 中间舍入，所以这 5 份配置的动作输出和误差相同，差别只在耗时，见性能页。表中的“逐位相同”是 8 核板上的结果；这些配置也可写 `rpu_execution.model.num_cores = 4 / 6`，4/6 核已上板通过运行时检查，并启用 4 核原生 Vision merger 与少核动作专家融合路径，但跨核归约顺序不同，因此输出与 8 核不逐位相同。各开关的含义和依赖见[运行时配置](../../../../../docs/runtime_config.md)。

## 准备参考输入后运行

准备与 v3 `steps_150000` non-RTC 配方匹配的源码、checkpoint 及 CPU fixture。验证脚本会核对资产身份；图像来自配套源码的示例 episode，状态和噪声为固定种子的合成探针，不用于评估机器人任务成功率。

从仓库根目录执行。以下 `/path/to/...` 均需替换为实际路径；若没有匹配 fixture，先用 `prepare` 阶段在独立进程中生成：

```bash
python scripts/validate_rhinovla_v3.py prepare \
  --source /path/to/rhinovla-v3/source \
  --checkpoint /path/to/rhinovla-v3/checkpoint \
  --fixture /path/to/rhinovla-v3/cpu_fixture.pt
```

复制所选 TOML，在 `[model]` 中填写 `source_dir` 和 `checkpoint`，在 `[input]` 中填写 `fixture`，然后运行：

```bash
python examples/run.py --config /path/to/filled-fp16.toml --check-config
python examples/run.py --config /path/to/filled-fp16.toml
```

`run.py` 使用 fixture 中第 1 组输入，预热 2 次、正式 5 次，计时口径与其他示例相同；结果目录见 `[run].output_dir`。

## 用原始图片运行

不用 fixture 时，删去 `[input].fixture`，改写下列字段（与 fixture 二选一，两者都写或都不写会被拒绝）。模型加载后由源码自己的 processor 和聊天模板构造 qin，与 `validate_v3.py reference` 生成 fixture 的方式相同；视角模态固定为 `rgb`。

| 字段 | 范围 | 省略时 |
|---|---|---|
| `images` | 1–3 个图片路径，顺序与 `view_roles` 对应；任意尺寸，转 RGB 后用 PIL 默认插值缩放到 256×256（源码训练配置的 `image_size`），每图 64 个图像 token | 必填 |
| `instruction` | 非空文本；prefix 行数 = 聊天模板与视角标签 + 每图 64 + 指令 token，由 processor 计算，加载时由运行时检查 | 必填 |
| `view_roles` | 每图一个，取源码训练配置 `view_role_vocab` 的值：`side`、`hand`、`top_head`、`hand_left`、`hand_right`、`head`、`head_center`、`head_left`、`head_right`、`head_back`、`back_left`、`back_right` | 取 `top_head`、`hand_left`、`hand_right` 的前 C 个（C 为图片数） |
| `state` | 1–96 个有限数值，补零到 96 维；`state_mask` 前 `len(state)` 维为 1 | 必填 |
| `action_dims` | 1–96 的整数；`action_mask` 前 `action_dims` 维为 1 | 等于 `len(state)` |

初始噪声 `x0[1,30,96]` 由 `[run].seed`（省略为 0）的 CPU 生成器产生，再乘 `action_mask`；种子 0 与 fixture 第 1 组的 `x0` 相同。报告的 `input_geometry` 记录图片数、视角、prefix 行数、图像网格和 qin 摘要。原始输入没有对应的 CPU fixture，所以不做 CPU 对比。训练用的是 3 个视角，1/2 图属于训练分布之外的输入。原始图片入口的 FP16 1/2/3 图 8 核和 1 图 6 核已上板通过运行时检查（2026-11-07）。3 图时的提示文本、230 个 prefix token 和图像网格与 fixture 第 1 组相同，但像素值最多差 3 个灰度级：图取自示例 episode 视频首帧，解码结果与生成 fixture 时的帧有细小差别，所以 qin 摘要不同。

需要同时核对运行时正确性（与 CPU fixture 对比、A/B/A 输入切换、Graph 稳定回放）时，用验证脚本运行同一文件；报告目录需预先存在：

```bash
python scripts/validate_rhinovla_v3.py run \
  --source /path/to/rhinovla-v3/source \
  --checkpoint /path/to/rhinovla-v3/checkpoint \
  --fixture /path/to/rhinovla-v3/cpu_fixture.pt \
  --config /path/to/filled-fp16.toml \
  --output /path/to/results/rhinovla-v3-actions.pt
```

脚本读取 `[model.opt_in]` 中的开关，命令行的 `--source`、`--checkpoint`、`--fixture` 须与 TOML 中填写的一致。脚本输出 `RUNTIME_PASS` 或失败信息，并保存 JSON 报告和动作张量。该结果描述本次运行；数值诊断、任务质量及支持状态需分别判断。使用自有模型集成时，改用上一级的[调用者工厂模板](../README.md)。
