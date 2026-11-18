# RhinoVLA v3 调用者工厂配置

<a id="rhinovla-v3-配置"></a>

[模型页](../README.md) · [模型配置表](../README.md#configurations) · [核数支持矩阵](../../../../docs/planner_core_topology.zh.md) · [权重准备](../../WEIGHTS.md)

当前目录的 TOML 通过 `examples/run.py` 调用你提供的 runtime factory。支持状态见[模型配置表](../README.md#configurations)。`controlled/` 中的配置使用 v3 源码入口，可直接用 `examples/run.py` 运行，也可交给 `validate_v3.py` 校验。

<a id="配置矩阵"></a>

| 配方 | 配置 |
|---|---|
| FP16 | [fp16.toml](fp16.toml) |
| W8A16 Action | [w8a16_action.toml](w8a16_action.toml) |
| W8A16 Action，含 AdaRMS | [w8a16_action_adarms.toml](w8a16_action_adarms.toml) |
| W8A16 pipeline | [w8a16_pipeline.toml](w8a16_pipeline.toml) |
| W8A16 pipeline，高精度融合与 vector Q/K | [w8a16_pipeline_high_fused_vector_qk.toml](w8a16_pipeline_high_fused_vector_qk.toml) |

## 接入工厂并运行

填写 `model.checkpoint`、`input.request` 和 `model.factory_config`。将 `model.runtime_factory` 的 `your_package...` 占位符替换为实际的 `module:callable`，该工厂必须提供所选精度的模型。保留 `trust_model_code = true` 和 `num_steps = 10`；factory config 中的量化选项必须与 TOML 一致。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/rhinovla/v3/fp16.toml --check-config
python examples/run.py --config examples/configs/rhinovla/v3/fp16.toml
```

配置检查只验证字段与配方关系；工厂能否加载、实际安装的量化权重和输入内容在运行时检查。使用独立验证脚本时，改看 [controlled 指南](controlled/README.md)。

`input.request` 是 `torch.save` 保存的 tensor 字典，原样作为 `factory.predict(**request)` 的参数，格式由你的 factory 决定；`model.factory_config` 是传给 factory 的 JSON。仓库内没有生成这两个文件的脚本。

<a id="execution-switches"></a>

## 执行开关

本目录 5 个模板的 `[rpu_execution.components.*]` 写出了下列冷启动布尔开关，全部为 `true`。这些开关在创建 runtime 时一次绑定，改值须新建进程。字段省略时读取同名的旧环境变量，环境变量也未设置时为 `false`（对应关系见[运行时配置](../../../../docs/runtime_config.md)）。`controlled/` 中的配置不写这些字段，改由 `[model.opt_in]` 中的同名环境变量设置。

| 组件 | 字段 | 作用 | 依赖 |
|---|---|---|---|
| `vision_encoder.vision` | `rpu_patch_embed` | patch embedding 在 RPU 上执行 | 全流程 W8 Vision 必须开启 |
| `vision_encoder.vision` | `rpu_mergers` | Vision merger 在 RPU 上执行 | 全流程 W8 Vision 必须开启 |
| `vision_encoder.vision` | `fused_merger` | 把 4 个 merger MLP 并入 Vision 的设备段，去掉 Vision 到 prefill 之间的 eager Linear | — |
| `vision_encoder.vision` | `bake_merger` | 把融合后的 merger 固化进 BUILD 的 Graph，REPLAY 时不再由主机重新发出约 370 个节点 | `fast_replay`、`fused_merger` |
| `vision_encoder.vision`、`language_model.prefill` | `preload_replay_skip` | REPLAY 时跳过主机对 preload 回调的重新遍历（权重/norm 的 DMA 与 memset） | 只在 REPLAY 时起作用 |
| 三个组件 | `fast_replay` | Graph 快速回放，见[通用参数](../../PARAMETERS.md#模型专用参数) | — |
| `language_model.prefill` | `prefix_prep_cache` | 缓存 prefix 中不随帧变化的输入（token、mask、图像 grid）；之后这些输入改变时报错，须新建 runtime | — |
| `language_model.prefill` | `prefix_prep_rpu` | 用上述缓存在 RPU 上组装 prefix embedding | `prefix_prep_cache` |
| `language_model.prefill` | `prefix_no_clone` | 不再为每次请求复制 merger 输出槽和 prefix 基础缓冲 | — |
| `language_model.prefill` | `partial_mrope` | 使用 prefix 缓存中预先计算的 M-RoPE 表 | `prefix_prep_cache` |
| `language_model.prefill` | `merger_scatter` | Vision merger 直接写入 prefix 的常驻缓冲 | `prefix_prep_rpu`、`fused_merger` |
| `action_expert.action` | `shared_cache` | prefix KV 留在 RPU 文本 cache 中交给 action，不导出到 CPU | — |
| `action_expert.action` | `prefix_alias` | action expert 直接把文本 cache 的 prefix 层用作 KV 存储，省去一次复制 | `shared_cache` |
| `action_expert.action` | `denoise_unroll` | 展开去噪循环，预先计算每步的 AdaRMS 与时间投影表 | — |

依赖不满足时，配置在安装权重前报错。代码注释中记录 `bake_merger`、`preload_replay_skip`、`prefix_no_clone`、`merger_scatter` 的开关前后结果逐位相同；其余开关没有单独的数值记录，各配置的误差见性能报告。`rpu_mergers_streaming` 模板未写，省略为 `false`，开启时须同时开启 `rpu_mergers`。

<a id="rtc"></a>

### RTC：FP16 与 W8A16，delay 0–6

RTC 绑定上述固定 v3 源码与 checkpoint、batch 1、30×96 动作和 18 层专家，使用十步展开去噪。RTC 是与精度配方独立的运行模式，不要求权重必须为 FP16；下面五份模板与对应 non-RTC 配置具有相同的量化范围和优化设置。

| RTC 配置 | 对应的精度配方 |
|---|---|
| [rtc_fp16.toml](rtc_fp16.toml) | FP16 |
| [rtc_w8a16_action.toml](rtc_w8a16_action.toml) | 专家七类主干投影 W8 |
| [rtc_w8a16_action_adarms.toml](rtc_w8a16_action_adarms.toml) | 专家主干与 AdaRMS 条件投影 W8 |
| [rtc_w8a16_pipeline.toml](rtc_w8a16_pipeline.toml) | 全流水线 W8 |
| [rtc_w8a16_pipeline_high_fused_vector_qk.toml](rtc_w8a16_pipeline_high_fused_vector_qk.toml) | 全流水线 W8 + HIGH/fused/vector QK |

配置登记均为 `example_only` 模板，要求使用内置 v3 factory。RTC profile 在原精度 profile 的 `rhinovla.v3.` 后增加 `rtc.`，例如 `rhinovla.v3.rtc.w8a16.expert.trusted-factory`。v1、其他 horizon/步数及 `delay > 6` 不在此契约内。RTC 验证需覆盖所选精度与配置，不继承 non-RTC 的测试结论。

继续使用[统一安装指南](../../../../docs/getting_started.md)创建的 Miniconda3 `RHINOFORGE_ENV`。完整环境包可提供 `vla` 依赖；运行 RTC 还需要包含本节接口的新版本 RhinoForge 源码及同版本编译的 native 扩展。依赖环境包可继续复用；更新 RhinoForge 源码后需按统一指南重新编译安装，不能仅复制 RTC Python 文件或 TOML 后沿用旧 native 库。按照前文生成普通 `factory.json`、`qin.pt` 和 `request.pt`，再创建 RTC 输入：

```bash
export RHINOVLA_V3_EXAMPLE=/tmp/rhinovla_v3_example
python - <<'PY_RTC'
import json
import os
from pathlib import Path
import torch

root = Path(os.environ["RHINOVLA_V3_EXAMPLE"])
factory = json.loads((root / "factory.json").read_text())
factory["rtc"] = True
(root / "rtc_factory.json").write_text(json.dumps(factory, indent=2))
request = torch.load(root / "request.pt", map_location="cpu", weights_only=True)
request["delay"] = 6
# A deterministic test prefix. In deployment, use the already committed actions
# in the same normalized 96D action space as this checkpoint.
request["action_prefix"] = torch.linspace(-0.25, 0.25, 6 * 96).reshape(1, 6, 96)
torch.save(request, root / "rtc_request.pt")
PY_RTC
cp examples/configs/rhinovla/v3/rtc_fp16.toml /tmp/rhinovla_v3_rtc.toml
# Fill input.checkpoint, input.factory_config, input.request and run.output_dir.
vi /tmp/rhinovla_v3_rtc.toml
python -m examples.run --config /tmp/rhinovla_v3_rtc.toml --check-config
python -m examples.run --config /tmp/rhinovla_v3_rtc.toml
```

运行 W8 档位时，将复制的模板替换为上表对应文件，继续使用同一份 checkpoint、`rtc_factory.json`、`qin.pt` 和 `rtc_request.pt`。保留模板的 `dtype`、`expert_w8a16/full_w8a16` 与环境开关；如果 factory JSON 也包含量化布尔值，其值必须与模板一致。runner 会检查实际安装的专家权重和 scale，全流水线模板还会检查 Vision、Text、动作 IO 与 AdaRMS 的量化存储。

`--check-config`只检查 TOML；实际运行时，runner 会在加载 qin 和模型权重前检查 RTC 前缀与 delay。内置 factory 固定为 `rpu_backend.adapters.rhinovla.factory:v3_runtime_factory`。`factory_config.rtc=true` 是构造时设置；更改它需要重建 policy。

| 字段 | 契约 |
|---|---|
| `factory.json` 的 `rtc` | RTC 模板要求 JSON 布尔值 `true`；non-RTC 模板保持省略或 `false` |
| `request["delay"]` | Python `int`，允许 `0,1,2,3,4,5,6`；不接受 `bool`、浮点数或 Tensor |
| `request["action_prefix"]` | 有限的 CPU 浮点 Tensor，形状 `[1,D,96]` 或 `[D,96]`；`delay <= D <= 30` |
| `x0/state/state_mask/action_mask` | 有限的 CPU 浮点输入；动作掩码在 30 行间保持相同；输出为 `[1,30,96]` |

RTC 请求必须同时提供 `action_prefix` 和 `delay`。`delay=0` 时仍需提供 prefix，可用 `torch.empty(1,0,96)`；前缀中第 `delay` 行之后的值不参与固定区间。掩码处理后的前缀还须能表示为有限 FP16。普通 non-RTC policy 不接收这两个参数。

RTC 每一步先复写已确定的前 `delay` 个动作，再执行去噪；这些 token 的时间为 0，其余动作使用本步时间。state token 使用全部动作时间嵌入的均值，最终输出再次固定前缀并应用动作掩码。只在输出阶段覆盖历史动作不能实现这一计算路径。

RTC 模板保留对应 non-RTC 配方的 `PRECOMPUTE_ADARMS=1`、`EXPERT_FUSIONS=1`、`ADARMS_RESIDENT=1` 和 `denoise_unroll=true`，W8/HIGH/vector 开关随所选配方保持一致；`PACKED_QKV` 与 `SKIP_ADARMS_GEMV` 继续关闭。RTC 会为不同 delay 准备匹配的条件参数，不能复用 non-RTC 的统一时间表。使用模板中的整组设置。每个 delay 对应独立的动作 Graph 签名；同一 delay 更新 prefix 内容需要刷新输入。runtime 在构造时自动预留 10 个 Graph arena，覆盖七个 delay 的动作 Graph、Vision/Language 保留 Graph 和一个准备阶段的临时 Graph；用户无需额外设置环境变量。固定 Graph 集合前先预热需要使用的所有 delay。验证应逐个覆盖 0–6、同 delay 前缀变化、0→6→0 切换和输出存储独立性，不能用 non-RTC 的既有结果替代。

启用 `ADARMS_RESIDENT` 时，RTC 每个去噪步将该步的 state、已确定 prefix、剩余 suffix 三组条件表装入 SPM，在该步 18 层专家中复用，为前序 Vision/Text 保留缓冲区留出空间。non-RTC 仍驻留十步条件表；用户保留模板中的驻留与融合开关即可。

示例输入使用三幅等尺寸视图与 230 个逻辑 prefix token（P230）。模板会在构造时固定 qin 中的 token、attention mask 和 grid；这些内容或像素几何变化时需新建 runtime，同几何的像素值可更新。异尺寸视图不在这些模板的输入范围内，强制 oneshot 模式会被提前拒绝。

<a id="parameters"></a>

### 需要填写和常用调整项

| 字段 | 模板设置与调整方法 |
|---|---|
| `input.num_steps` | 固定 `10`，AdaRMS 表与十步 schedule 配套 |
| `input.expert_w8a16/full_w8a16` | 按所选配方保留；同时匹配 `example.profile_id`、`dtype` 和 factory JSON |
| `rpu_execution.components.language_model.prefill.linear_acc32` | 全流水线模板显式为 `false`；`true` 选择 Text Linear ACC32 |
| `rpu_execution.components.vision_encoder.vision.linear_acc32` | 全流水线模板显式为 `false`；`true` 选择 Vision block 与融合 merger 的 Linear ACC32 |
| `rpu_execution.components.language_model.prefill.padding_rows` | 全流水线为 `10`；随实际 prefix 的物理执行范围调整 |
| `cpu.threads` | 全流水线为 `12`，其余继承环境 |
| `run.warmup/runs` | 默认 `2/3` |

公共字段见[参数说明](../../PARAMETERS.md)。

Text/Vision 的 ACC 字段不控制 action、patch embedding 和冷 AdaRMS 投影，也不写入 action 表。现有优化要求完整 18 层专家、96D 动作和上述十步输入结构。

### 实现参考

- [runner](../../../runners/rhinovla/runner.py)：绑定 factory 参数并核对实际量化范围。
- [convert.py](../../../../python/rpu_backend/adapters/rhinovla/convert.py)：专家投影量化、AdaRMS 预计算及动作 IO 安装。
- [precision.py](../../../../python/rpu_backend/adapters/rhinovla/precision.py)：Text 量化、Linear 累加与全流水线权重检查。
- [pipeline.py](../../../../python/rpu_backend/adapters/rhinovla/pipeline.py)：组件构造、缓存和执行。
- [运行参数](../../../../docs/runtime_config.md)与[两份运行库](../../../../docs/runtime_assets.md)。
