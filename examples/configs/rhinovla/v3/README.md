# RhinoVLA v3 factory 模板

本目录提供版本明确的公开运行入口：调用者的 v3 factory 接收 checkpoint、十步 denoise 设置、factory JSON 和冷配置，返回 `RhinoVLAPolicy.from_factory` 接受的 runtime。示例不附带 v3 模型源码、checkpoint、输入数据或数值基准。

| 配置 | `expert_w8a16` | `full_w8a16` | 选择的精度范围 |
|---|---|---|---|
| `fp16.toml` | 未传入，默认关闭 | 未传入，默认关闭 | FP16，保留既有 factory 参数兼容 |
| `expert_w8a16.toml`（兼容名 `w8a16.toml`） | `true` | `false` | expert 每层 Q/K/V/O、gate/up/down 七类投影使用 INT8 权重、FP16 scales/激活 |
| `full_expert_w8a16.toml` | `true` | `true` | 七类投影及 expert 每层 AdaRMS 条件投影使用 W8A16 |
| `full_pipeline_w8a16.toml` | `true` | `true` | 完整流水线 W8；显式关闭 HIGH、gate-TANH 预计算与 vector Q/K norm |
| `full_pipeline_w8a16_high_precision_vector_qk.toml` | `true` | `true` | 另显式启用全流水线 W8、HIGH 非线性/归一化与 Q/K norm 选项 |

前两种 expert W8 配置中的 `dtype="w8a16"` 描述所选 **action expert** 范围，均不选择 Vision、Language prefix 或 action IO 的 W8 量化。`full_expert_w8a16` 不是全模型/全流水线 W8；其中 `full_w8a16` 名称沿用公开 `build_rpu_expert` 的参数，只在 expert 构造范围内解释。模型版本、其余组件精度、输入维度及 denoise schedule 仍须由调用者的集成明确校验。

`--check-config` 会要求 W8 模板同时给出两个严格布尔值，且 `expert_w8a16=true`。构造前，runner 将它们写入传给 factory 的配置映射；factory JSON 中已有同名值必须类型和值均一致，FP16 也不能在 JSON 中偷偷开启这些 W8 选项。没有新增字段的原 FP16 factory 不会收到额外的参数。

factory 在读取模型权重之前应校验这些选择，构造 expert 时直接复用已有 API，例如以下片段（不构成完整 pipeline）：

```python
from rpu_backend.adapters.rhinovla import build_rpu_expert

expert = build_rpu_expert(
    cpu_expert,
    prefix_len=prefix_len,
    suffix_len=suffix_len,
    max_seq_len=max_seq_len,
    w8a16=config.get("expert_w8a16", False),
    full_w8a16=config.get("full_w8a16", False),
)
```

对于 expert-only 配置，使用公开 `RhinoVLAOnRPU` orchestrator 的 factory 应将两个选择传入它的冷安装参数，而不设置全流水线环境开关：

```python
from rpu_backend.adapters.rhinovla.pipeline import RhinoVLAOnRPU

runtime = RhinoVLAOnRPU(
    qin=qin, prefix_len=prefix_len, steps=10,
    model=model, action_bundle=action_bundle,
    flow_direction="official_descending", rpu_execution=rpu_execution,
    expert_w8a16=config.get("expert_w8a16", False),
    full_expert_w8a16=config.get("full_w8a16", False),
)
```

`full_expert_w8a16=True` 使用 18 组实际 W8 条件投影生成 AdaRMS 表，最终 norm 的条件投影保持浮点预计算，最终表和 action IO 保持 FP16。其冷身份独立于 IO 精度。既有 `RPU_RHINOVLA_FULL_W8A16=1` 全流水线路径仍要求 19 组 W8 条件投影 owner 和 6 个 IO scale；HIGH/TANH 预计算仍仅属于该路径，不能用于 expert-only 配置。显式参数与环境精度冲突会在权重安装前拒绝。

随后用 `bind_rhinovla_execution_runtime(..., action_expert=expert, ...)` 绑定这个实际 expert 及模型的 text/vision 子模块。runner 在首次推理前会核对 controller 与 runtime 的所有权、七类投影实际 INT8/FP16-scale 存储、native 安装时记录的 expert 模式；full-expert 模板还检查实际 AdaRMS INT8 权重和 FP16 scales。factory 忽略选择或返回其他精度时会报错并关闭已创建的 policy。这不是 CPU 数值验证，也不能证明调用者自定义 pipeline 的全部计算精度。

W8 构造使用公开 expert 的 v3 维度约束；不兼容 packed QKV。两模板固定 `RPU_RHINOVLA_PACKED_QKV=0`。full-expert 还固定 `RPU_RHINOVLA_PRECOMPUTE_ADARMS=1`、`RPU_RHINOVLA_SKIP_ADARMS_GEMV=0`：factory 必须为实际 timestep schedule 准备并安装匹配的 AdaRMS 表，不能只设置环境开关。权重变换不可重复，构造失败须从干净 checkpoint 重建。

模板保留现有公开 RhinoVLA runtime 的优化控制：

- Vision：RPU patch/merger、融合 merger、稳定 replay 与 preload、baked merger。
- Language：prefix preparation cache、RPU preparation、前缀存储复用、partial MRoPE、merger scatter 与 replay。
- Action：denoise unroll、共享 cache、prefix alias 与 replay。

FP16、expert W8 和 full-expert W8 模板的 `[example.opt_in]` 还明确开启 Vision view batching/preparation cache、denoise static-context cache、AdaRMS precompute/residency、gated residual 与 SiLU-multiply/expert fusions、aligned KV。这九项在直接 adapter 中默认关闭，不能由 `rpu_execution` 自动推导。这些模板固定全流水线/环境 expert W8 为 `0`，expert 精度只由上述严格布尔参数选择；packed QKV 和 skip-AdaRMS 也固定为 `0`。融合与缓存仍须通过同输入数值核对，模板不承诺某个历史耗时。

两个 `full_pipeline_w8a16*.toml` 均为实验配置，不代表默认推荐或已取得独立任务质量证据。它们同时显式设置两个 W8 环境选择为 `1`。其范围包括 text prefix 七类投影、vision block 投影、折叠 patch projection、所有 merger MLP、expert 七类投影、action IO，以及含 final norm 的冷 AdaRMS 投影；norm、bias、embedding 和位置张量仍为浮点。两者均启用 vision input cache，基础版显式关闭 HIGH、gate-TANH 预计算和 vector Q/K norm，`high_precision_vector_qk` 版则显式开启这些选项，便于在相同量化范围下比较算术与融合路径。HIGH **不代表** Linear ACC32。runner 对两者都会额外检查实际 text/vision/IO/AdaRMS 权重、scale 及绑定的三个子 owner，expert-only factory 不能满足这些配置。量化实现与同量化参考的误差、相对原 FP32 模型的质量变化应分别记录；能运行或更快不代表机器人任务质量通过。

完整流水线模板使用 12 个 Torch CPU 线程、`no_grad` 执行（`inference_mode=false`），一次 warmup 后计时三次。默认 text 和 vision `linear_acc32=false`（ACC16）。可分别在 `rpu_execution.components.language_model.prefill` 与 `rpu_execution.components.vision_encoder.vision` 中将其设为 `true`，选择现有 Linear 的 ACC32 累加；vision 包括 block 与融合 merger，patch embedding、action 和冷 AdaRMS 的累加策略不随之变化。这两个严格布尔字段冷绑定，不能运行中更改，也不能放到 action 组件中。模板中的 `padding_rows=10` 是物理 prefix padding，调用者需按自己的逻辑输入与有效执行范围调整，不改变真实 token 数。

组件控制写在对应 `rpu_execution.components` 中，在构造前生效。factory 必须绑定 `RhinoVLAPipelineColdConfig` 并声明实际支持；不具备相应实现时应拒绝配置。公开 runtime 会检查依赖关系，例如 `prefix_prep_rpu` 需要 `prefix_prep_cache`，`merger_scatter` 需要 RPU prefix preparation 和 fused merger，`bake_merger` 需要 fast replay 和 fused merger，`prefix_alias` 需要 shared cache。输入内容变化时仍必须刷新可变状态，不能把不同请求当作固定缓存。

这些模板沿用同一组公开冷优化控制；具体能力仍由集成方按实际 pipeline 绑定，不能因为选择 W8 就假定所有优化已实现。额外环境控制参见 [公开 RhinoVLA runtime 配置](../../../../docs/runtime_config.md#rhinovla-profile)。

运行和数值核对应使用同一版本 factory 的真实请求及 CPU 输出。`--check-config` 只检查配置结构；本目录没有复制私有验证脚本、输入 fixture、报告或其身份信息。

## 数值评估

采用三层结果，不把任意 FP32 误差限值当作所有精度通用的失败门：

1. 语义/配置、shape/dtype、finite、输入刷新、缓存和 Graph 生命周期属于运行时硬检查。同一确定性运行时的重放合同与跨参考数值比较分开。
2. cosine、逐行分布、MSE/MAE、relative L2 和最大误差完整记录。未独立校准的阈值不决定 PASS/FAIL；相同参考输出也不要求逐字节一致。量化实现等价应与相同量化权重和算术语义的参考比较；对未量化 FP32 原模型的差异还包含量化影响，不能混为实现错误。
3. 反归一化动作、轨迹或闭环任务质量需要适用的独立证据和校准合同；没有证据时标为 `NOT_EVALUATED`，不能因删除误差门就宣称任务通过。

旧报告保留当时测量值与判词，属于历史记录，不再作为当前自动失败条件。运行时通过、浮点观测和任务质量必须分别报告；本次策略调整不改变精度范围、输入语义、执行安全约束或历史性能样本。
