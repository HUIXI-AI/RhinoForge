# 通用配置参数

[模型目录](README.md) · [权重准备](WEIGHTS.md) · [命名规则](NAMING.md) · [日志与结果](../RESULTS.md#日志设置)

复制对应模型的完整 TOML，再填写路径和输入。模型页列出该模板的实际设置及可选范围；本页解释公共字段。模板中省略的字段由运行器或环境补全，因此“模板未写”和“关闭”分别理解。

## 配置身份与权重

| 字段 | 用途 | 设置方法 |
|---|---|---|
| `example.profile_id` | 选择登记的模型结构、精度和执行路径 | 随完整模板选择；部分配方通过它区分输出头量化、核数或加载方式 |
| `example.registry_alias` | 定位缓存中的模型资产 | 与 profile 配套，目录由模型登记表解析；设置 `RPU_MODEL_CACHE` 指定缓存根目录 |
| `example.dtype` | 登记配方的精度标签 | 与 profile、资产格式一致；具体模块精度见模型页 |
| `example.target/architecture` | 选择运行器与模型架构 | 使用模板已有值 |
| `example.opt_in` | 模型安装及执行开关 | 按该配方保留；单项调整的条件在模型页说明 |
| `input.checkpoint` 等资产字段 | 显式指定权重或配套文件 | 仅在接受该字段的模型中使用；填写模型页下载/转换步骤得到的路径 |
| `pi05.precision` 等模型字段 | 模型专用入口的配方选择 | 与对应权重目录配套，直接选择相应模板 |

不同模型使用的资产字段不同。Qwen 系列通过 `registry_alias` 定位目录；Pi0.5 等还接受 checkpoint 绝对路径。多数入口填写实际路径；LingBot v2 等支持环境变量展开的入口按其模板设置。TOML 本身不执行 Shell 替换。

文件名便于选型，实际加载由 TOML 内容决定。切换配方时同时匹配权重、profile、精度和组件设置；选择完整模板可以保留这些关联。

## 输入

| 字段 | 含义 | 关联要求 |
|---|---|---|
| `input.prompt/images` | 提示词和图片路径 | 图片数量、分辨率和 processor 与模型配套 |
| `input.prefill_tokens` | 目标 prefill token 数 P | 文本的补齐方式、图文 token 计算方法由模型页说明 |
| `input.decode_steps` | prefill 后的 decode 调用上限 D | prefill 已产生第一个输出 token，执行满 D 次时输出 D+1 个 |
| `input.stop_on_eos` | 生成 EOS 后是否提前结束 | 省略行为由模型运行器决定；固定负载模板通常显式关闭 |
| `input.batch` | 保存预处理 tensor 字典的文件路径 | 匹配相机槽位、图像 shape、状态、token 和 mask |
| `input.cameras` | 相机槽位数 | 与输入 batch 和模型槽位定义匹配 |
| `input.text_tokens` | 模型定义的文本长度 | Pi0.5 等使用 token tensor 宽度，GR00T 使用 instruction 有效 token 数，范围见模型页 |
| `input.noise/num_steps` | 动作模型的初始噪声和去噪步数 | 与动作 shape、schedule 及预计算表配套 |

并非每个模型都接受以上全部字段。完整输入结构见所选模型页。配置检查核对结构和登记关系，资产和 tensor 形状在运行时检查。

## 执行与采样

| 字段 | 用途 | 设置方法 |
|---|---|---|
| `cpu.threads` | 请求 CPU 线程数 | 正整数；省略时继承运行器或环境，具体有效值见模型页和运行摘要 |
| `rpu_execution.model.num_cores` |设备并行根核数 TP | 使用该模型已提供的核数配置；部分模型的各组件实际核数不同 |
| `rpu_execution.*.linear_acc32` | 对应组件 Linear 的累加精度 | `false` 为 ACC16，`true` 为 ACC32；不改变权重文件的位宽 |
| `rpu_execution.*.chunk_size` | 对应组件执行分块 | `auto` 由执行器选择；整数范围取决于模型和输入 |
| `rpu_execution.*.padding_rows` | 物理执行补齐行数 | 与实际输入和布局匹配，补齐行不增加有效 token |
| `run.warmup` | 预热次数 | 公共登记入口省略为 `2`；HyVLA 至少 `3`；LingBot v2 允许 `0`，仍执行准备阶段；其余按模型页要求；预热不计入正式统计 |
| `run.runs` | 正式运行次数 | 公共登记入口省略为 `1`，取正整数；专用入口和模板的次数见模型页 |
| `run.seed` | 随机种子 | 公共登记入口省略为 `0`，取非负整数；各入口按其规则生成输入或噪声 |
| `run.inference_mode` | Torch 推理上下文 | 公共入口省略时为 `true`；`false` 使用 `no_grad`，专用入口按模型页说明 |
| `run.output_dir` | 运行摘要与产物目录 | 公共登记入口省略为 `perf_results/examples`；专用入口可能要求填写，子目录和文件名见模型页 |
| `run.profile` | 额外分析调用 | 按模型入口启用，额外调用与正式推理样本分开查看 |
| `run.hwperf` | 硬件跟踪采集 | 采集范围由运行器决定，可能覆盖加载、预热和正式循环；具体范围见模型页 |
| `run.timing_mode/summary_statistic` | 指定测量边界与统计方式 | 按模型运行器支持的边界与统计方式设置 |

表中的 `*` 表示模板已有的组件位置，不是可直接填写的 TOML 字段。多数 LLM 使用 `prefill`、`vision`；RhinoVLA factory 使用 `components.language_model.prefill` 等路径，完整执行配置见[运行时配置参考](../../docs/runtime_config.md)。

CPU 线程与设备核数分别设置。权重精度、累加精度和非线性算法也分别控制；低位宽权重不等于低精度累加。权重、核数、累加和安装开关变更后，用新进程重新加载。

## 日志与配置检查

`logging.level` 默认为 `ERROR`，`logging.summary` 默认为 `true`。取值、终端显示和日志文件见[日志与结果](../RESULTS.md#日志设置)。

```bash
python examples/run.py --config /path/to/filled-config.toml --check-config
python examples/run.py --config /path/to/filled-config.toml
```

第一条命令检查字段和登记关系，第二条命令加载资产并运行。专用验证片段使用模型页标出的入口。
