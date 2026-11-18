# RhinoVLA v3 源码入口

<a id="rhinovla-v3-受控配置"></a>

[v3 模型页](../README.md) · [模型配置表](../../README.md#configurations) · [权重准备](../../../WEIGHTS.md)

本目录的 TOML 直接从 v3 `steps_150000` non-RTC 源码、checkpoint 和 CPU fixture 构建模型，用 `examples/run.py` 运行。每份文件对应一组固定的冷启动开关（`[model.opt_in]`），与登记的 example-only profile 完全一致，修改开关会被拒绝。20 个开关全部写出，关闭的写为 `"0"`；shell 中继承的同名环境变量若与之不同，运行时直接报错。精度同时显式传给 factory，构建后再检查已安装的 expert 权重，与 `validate_v3.py` 的检查一致。同一文件也可交给 `validate_v3.py --config` 做运行时校验。

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

high 之后的 4 个开关只改变算子融合、Q/K 归一化的向量化方式和 RoPE 的实现，保留 HIGH 的计算和 FP16 中间舍入，所以这 5 份配置的动作输出和误差相同，差别只在耗时，见性能页。各开关的含义和依赖见[运行时配置](../../../../../docs/runtime_config.md)。

## 准备参考输入后运行

准备与 v3 `steps_150000` non-RTC 配方匹配的源码、checkpoint 及 CPU fixture。验证脚本会核对资产身份；图像来自配套源码的示例 episode，状态和噪声为固定种子的合成探针，不用于评估机器人任务成功率。

从仓库根目录执行。以下 `/path/to/...` 均需替换为实际路径；若没有匹配 fixture，先用 `reference` 阶段在独立进程中生成：

```bash
python python/rpu_backend/tests/model/rhinovla/validate_v3.py reference \
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

需要同时核对运行时正确性（与 CPU fixture 对比、A/B/A 输入切换、Graph 稳定回放）时，用验证脚本运行同一文件；报告目录需预先存在：

```bash
python python/rpu_backend/tests/model/rhinovla/validate_v3.py rpu \
  --source /path/to/rhinovla-v3/source \
  --checkpoint /path/to/rhinovla-v3/checkpoint \
  --fixture /path/to/rhinovla-v3/cpu_fixture.pt \
  --config /path/to/filled-fp16.toml \
  --report /path/to/results/rhinovla-v3.json
```

脚本读取 `[model.opt_in]` 中的开关，命令行的 `--source`、`--checkpoint`、`--fixture` 须与 TOML 中填写的一致。脚本输出 `RUNTIME_PASS` 或失败信息，并保存 JSON 报告和动作张量。该结果描述本次运行；数值诊断、任务质量及支持状态需分别判断。使用自有模型集成时，改用上一级的[调用者工厂模板](../README.md)。
