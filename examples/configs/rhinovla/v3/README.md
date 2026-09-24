# RhinoVLA v3 factory 模板

本目录提供版本明确的公开运行入口：调用者的 v3 factory 接收 checkpoint、十步 denoise 设置、factory JSON 和冷配置，返回 `RhinoVLAPolicy.from_factory` 接受的 runtime。示例没有内置 v3 checkpoint 装载器、完整 pipeline 或数值基准。

| 配置 | `expert_w8a16` | `full_w8a16` | 选择的精度范围 |
|---|---|---|---|
| `fp16.toml` | 未传入，默认关闭 | 未传入，默认关闭 | FP16，保留既有 factory 参数兼容 |
| `w8a16.toml` | `true` | `false` | expert 每层 Q/K/V/O、gate/up/down 七类投影使用 INT8 权重、FP16 scales/激活 |
| `full_expert_w8a16.toml` | `true` | `true` | 七类投影及 expert 每层 AdaRMS 条件投影使用 W8A16 |

两种 W8 配置中的 `dtype="w8a16"` 描述所选 **action expert** 范围，均不选择 Vision、Language prefix 或 action IO 的 W8 量化。`full_expert_w8a16` 不是全模型/全流水线 W8；其中 `full_w8a16` 名称沿用公开 `build_rpu_expert` 的参数，只在 expert 构造范围内解释。模型版本、其余组件精度、输入维度及 denoise schedule 仍须由调用者的集成明确校验。

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

随后用 `bind_rhinovla_execution_runtime(..., action_expert=expert, ...)` 绑定这个实际 expert 及模型的 text/vision 子模块。runner 在首次推理前会核对 controller 与 runtime 的所有权、七类投影实际 INT8/FP16-scale 存储、native 安装时记录的 expert 模式；full-expert 模板还检查实际 AdaRMS INT8 权重和 FP16 scales。factory 忽略选择或返回其他精度时会报错并关闭已创建的 policy。这不是 CPU 数值验证，也不能证明调用者自定义 pipeline 的全部计算精度。

W8 构造使用公开 expert 的 v3 维度约束；不兼容 packed QKV。两模板固定 `RPU_RHINOVLA_PACKED_QKV=0`。full-expert 还固定 `RPU_RHINOVLA_PRECOMPUTE_ADARMS=1`、`RPU_RHINOVLA_SKIP_ADARMS_GEMV=0`：factory 必须为实际 timestep schedule 准备并安装匹配的 AdaRMS 表，不能只设置环境开关。权重变换不可重复，构造失败须从干净 checkpoint 重建。

模板保留现有公开 RhinoVLA runtime 的优化控制：

- Vision：RPU patch/merger、融合 merger、稳定 replay 与 preload、baked merger。
- Language：prefix preparation cache、RPU preparation、前缀存储复用、partial MRoPE、merger scatter 与 replay。
- Action：denoise unroll、共享 cache、prefix alias 与 replay。

这些控制写在对应 `rpu_execution.components` 中，在构造前生效。factory 必须绑定 `RhinoVLAPipelineColdConfig` 并声明实际支持；不具备相应实现时应拒绝配置。公开 runtime 会检查依赖关系，例如 `prefix_prep_rpu` 需要 `prefix_prep_cache`，`merger_scatter` 需要 RPU prefix preparation 和 fused merger，`bake_merger` 需要 fast replay 和 fused merger，`prefix_alias` 需要 shared cache。输入内容变化时仍必须刷新可变状态，不能把不同请求当作固定缓存。

这些模板沿用同一组公开冷优化控制；具体能力仍由集成方按实际 pipeline 绑定，不能因为选择 W8 就假定所有优化已实现。额外环境控制参见 [公开 RhinoVLA runtime 配置](../../../../docs/runtime_config.md#rhinovla-profile)。

运行和数值核对应使用同一版本 factory 的真实请求及 CPU 输出。`--check-config` 只检查配置结构；本目录没有复制私有验证脚本、输入 fixture、报告或其身份信息。
