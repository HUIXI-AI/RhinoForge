# 权重准备

[模型目录](README.md) · [通用参数](PARAMETERS.md) · [公开配置索引](CONFIGURATIONS.md)

从模型页的配置表选择 TOML，并进入该行的权重准备步骤。每个步骤给出来源、下载或转换命令、最终目录及其配置字段。同一份权重可由多个输入或执行配置复用，模型页会明确列出适用范围。

<a id="quantization"></a>

## 从配置到运行资产

1. 确定模型版本、尺寸和任务，选择完整 TOML。
2. 打开该配置对应的获取步骤，下载指定原始模型、量化包，或准备交付资产。
3. 需要离线转换时，按步骤指定的源目录转换；加载时量化的配置直接使用源目录。
4. 将最终目录绑定到 `input.checkpoint` 等字段，或放到 `registry_alias` 对应的缓存目录。
5. 准备匹配的 tokenizer、processor、输入与输出目录，再检查配置并运行。

**配置比较顺序与权重转换顺序分别说明。** 例如 Pi0.5 的动作 NVFP4 相对 W8 解释计算差异，转换时仍读取原始浮点模型；预填充 W8A8 配置直接复用动作 NVFP4 产物。

下载工具使用自己的缓存规则；`RPU_MODEL_CACHE` 是本仓库查找已准备资产的根目录。registry 目录和转换输出按模型页填写，TOML 文件名不直接决定权重目录。

<a id="precision-notation"></a>

## 精度字段如何阅读

| 概念 | 含义 | 例子 |
|---|---|---|
| 原始存储精度 | 下载文件中各张量的 dtype | BF16、FP32，或两者混合 |
| 转换产物格式 | 离线脚本保存的量化权重、scale 和布局 | INT8 权重、打包 INT4、NVFP4 |
| 加载后的模块精度 | Vision、Text、Action、输出头等实际使用的权重和激活精度 | 部分 Linear 为 W8A16，其余模块为 FP16 |
| 累加精度 | Linear 等算子内部的累加方式 | ACC16、ACC32，由相应组件字段选择 |
| 非线性精度 | norm、SiLU、GELU、tanh 等的算术实现 | RhinoVLA 的 HIGH 开关 |

`W8A16` 表示所选投影的权重为 8 位、输入激活为 16 位；`G32/G128` 表示量化分组大小。INT4 可以打包存入 INT32/UINT8 容器，文件 dtype 与逻辑位宽分别查看。NVFP4 使用四位浮点及配套 scale，INT4 使用整数编码，各自对应不同的转换和加载路径。

例如，Pi0.5 的 W8 离线包仍保留浮点 Vision，运行 W8 配置时再量化指定 Vision 投影；RhinoVLA v3 的 HIGH 配置保持 W8 权重与 Linear ACC16，调整非线性、归一化、融合及 gate tanh 预计算。

## 配置怎样对应资产

| 关系 | 准备方式 | 配置指向 |
|---|---|---|
| 浮点直接加载 | 下载指定模型和版本，加载时转换浮点 dtype | 原始权重目录 |
| 加载时量化 | 下载指定浮点权重，由运行入口按 TOML 转换 | 原始权重目录 |
| 离线量化 | 执行模型页的转换命令，生成独立目录；某些模块在加载时继续转换 | 转换产物目录；需要浮点配套权重时另填对应字段 |
| 直接使用量化包 | 下载指定 AWQ/GPTQ 等包，按该配方加载 | 该量化包目录 |
| 客户交付资产 | 准备配套 checkpoint、源码、配置及输入 | 模型页列出的各个路径字段 |

权重之外还可能需要 tokenizer、processor、模型源码、factory 或预处理输入。模型页分别列出这些资产的用途和绑定位置。

<a id="sources"></a>
<a id="offline"></a>
<a id="runtime-quantization"></a>
<a id="no-local-converter"></a>

## 按模型查找

每个模型页集中维护下载来源、固定版本、模块精度、转换命令和路径绑定。

| 模型 | 原始资产与配方 | 配置及权重对应 | 下载、转换和运行 |
|---|---|---|---|
| <a id="cmd-qwen3"></a>Qwen3 | 对应尺寸的官方浮点权重；Instruct 使用单独模型；W8、W4 与输出头配方分别选择 | [对应表](qwen3/README.md#weight-recipes) | [操作步骤](qwen3/README.md#quantization) |
| <a id="cmd-qwen3-vl"></a>Qwen3-VL | 官方浮点权重或指定 AWQ 包；区分完整运行时配方和仅文本量化 | [对应表](qwen3_vl/README.md#weight-recipes) | [操作步骤](qwen3_vl/README.md#quantization) |
| <a id="cmd-qwen35-dense"></a><a id="cmd-qwen35-text-w4"></a>Qwen3.5 Dense | 官方 BF16/FP32 权重；统一 W8/W4、仅 MLP RTN、QuantTrio AWQ 分开 | [对应表](qwen3_5/README.md#weight-recipes) | [操作步骤](qwen3_5/README.md#quantization) |
| <a id="cmd-qwen35-moe"></a>Qwen3.5 35B-A3B MoE | 官方未量化浮点权重经独立脚本生成 RTN 或 mixed E4M3；官方 GPTQ 直接加载 | [对应表](qwen3_5/README.md#weight-recipes) | [RTN](qwen3_5/README.md#cmd-moe-rtn) / [GPTQ](qwen3_5/README.md#cmd-moe-gptq) / [mixed E4M3 下载与转换](qwen3_5/README.md#cmd-moe-mixed) |
| Llama 3.2-1B | Meta 官方 BF16 权重，加载为 FP16 | [对应表](llama/README.md#weight-recipes) | [操作步骤](llama/README.md#quantization) |
| Gemma4 E4B-it | Google 官方 BF16 权重，按组件加载 | [对应表](gemma4/README.md#weight-recipes) | [操作步骤](gemma4/README.md#quantization) |
| DINOv3 ViT-B | Meta 官方 FP32 权重，加载为 FP16 | [对应表](dinov3/README.md#weight-recipes) | [操作步骤](dinov3/README.md#quantization) |
| <a id="cmd-pi05"></a>Pi0.5 LIBERO | 指定 LeRobot LIBERO 微调权重；W8/NVFP4 离线转换，Prefill A8 共用 NVFP4 包 | [对应表](pi05/README.md#weight-recipes) | [操作步骤](pi05/README.md#quantization) |
| HyVLA | 腾讯 UMI 官方权重；FP16/W8/动作 NVFP4 共用浮点源目录 | [对应表](hy_vla/README.md#weight-recipes) | [操作步骤](hy_vla/README.md#quantization) |
| HY Embodied UMI | 腾讯 UMI 权重及该入口的配套输入 | [对应表](hy_embodied/README.md#weight-recipes) | [操作步骤](hy_embodied/README.md#quantization) |
| GR00T N1.7-3B | NVIDIA 策略权重和配套 Qwen 配置；加载时量化 | [对应表](gr00t/README.md#weight-recipes) | [操作步骤](gr00t/README.md#quantization) |
| RhinoVLA v1 | HuixiAI 固定 checkpoint、源码和 LIBERO fixture | [对应表](rhinovla/v1/README.md#weight-recipes) | [操作步骤](rhinovla/v1/README.md#quantization) |
| RhinoVLA v3 | 客户 96D 模型、factory 和请求；五份配置共用原始权重，加载时量化 | [对应表](rhinovla/v3/README.md#weight-recipes) | [操作步骤](rhinovla/v3/README.md#quantization) |
| <a id="cmd-wall"></a>Wall-OSS base | 官方 wall-oss-0.5 权重；W8/NVFP4 转换与浮点目录配对 | [对应表](wall_oss/base/README.md#weight-recipes) | [操作步骤](wall_oss/base/README.md#quantization) |
| <a id="cmd-internvla"></a>InternVLA-N1 NavDP | 官方模型及 System2 W8 配对包；System2 decoder W8，其余组件按混合精度执行 | [对应表](internvla/README.md#weight-recipes) | [操作步骤](internvla/README.md#quantization) |
| LingBot v2 | Robbyant 官方 FP32 权重；FP16/W8 加载时转换 | [对应表](lingbot/v2/README.md#weight-recipes) | [操作步骤](lingbot/v2/README.md#quantization) |
| LingBot v1 | 对应 4B 权重，CPU/设备混合执行 | [对应表](lingbot/README.md#weight-recipes) | [操作步骤](lingbot/README.md#quantization) |
| HALO 动作专家 | 配对 CoT EMA 权重和组件输入 | [对应表](halo/README.md#weight-recipes) | [操作步骤](halo/README.md#quantization) |
| <a id="cmd-legacy27"></a>Qwen3.8-27B 配套权重 | 独立浮点源与 W8 配方 | [对应表](qwen3_5/legacy/27b/README.md#配置说明) | [操作步骤](qwen3_5/legacy/27b/README.md#weights) |
| Wall-OSS legacy | 配套 checkpoint、模型源码与输入 | [对应表](wall_oss/legacy/README.md#weight-recipes) | [操作步骤](wall_oss/legacy/README.md#quantization) |
| LingBot exact-Z2 | 配套 6B checkpoint 与 Z2 输入 | [对应表](lingbot/legacy/exact_z2/README.md#weight-recipes) | [操作步骤](lingbot/legacy/exact_z2/README.md#quantization) |
