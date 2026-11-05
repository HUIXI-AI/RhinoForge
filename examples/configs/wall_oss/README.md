# Wall-OSS 配置

先按权重和输入选择模型组，再在组内选择精度。同为 W4 的配置可能量化不同模块，所需资产也不同；各组 README 负责具体转换与运行步骤。

<a id="选择配置"></a>

<a id="配置与权重对应"></a>

<a id="weight-recipes"></a>

## 配置说明

| 模型组 | 权重和输入 | 配方 | 配置及权重获取 |
|---|---|---|---|
| 公开基础模型 | Wall-OSS 0.5；两/三图，224/256；H32、十步 | FP16、W8、W8 + 动作 NVFP4 | [配置](base/README.md#配置说明) · [获取权重](base/README.md#准备与运行) |
| registry 输入 | 公开 Wall-OSS 0.5；图片、指令、状态和 mask；H32、十步 | FP16、W8、Language/Action 逐通道或 G32 INT4 | [配置](legacy/README.md#配置说明) · [获取权重](legacy/README.md#准备与运行) |

`H` 为动作长度。W8 使用 INT8 权重，W4 使用整数四位权重，NVFP4 使用 E2M1 四位浮点。同组内配置按下级指南复用源目录和量化目录。

<a id="准备权重并运行"></a>

<a id="quantization"></a>

<a id="weight-sources"></a>

<a id="weights"></a>

<a id="下载转换与运行"></a>

<a id="使用"></a>

## 准备与运行

具体来源、固定下载版本、转换命令和路径绑定集中在各组指南中：

<a id="download-parent"></a>

<a id="公开-base-w8a16"></a>

<a id="公开-base-w8--action-nvfp4"></a>

- base：[原始模型](base/README.md#download-parent)，[W8 与动作 NVFP4 共用配对目录](base/README.md#公开-base-w8a16)。

<a id="公开-base-legacy-w8a16"></a>

<a id="公开-base-legacy-w4逐通道-group0"></a>

<a id="公开-base-legacy-w4-pgrp混合-group32"></a>

- registry：[公开原始模型](legacy/README.md#download-parent)，[W8](legacy/README.md#公开-base-legacy-w8a16)，[逐通道 W4](legacy/README.md#公开-base-legacy-w4逐通道-group0)，[G32 W4](legacy/README.md#公开-base-legacy-w4-pgrp混合-group32)。

统一使用 `python examples/run.py --config <TOML>`。模型输入和常调参数在各组的“准备与运行”中，公共字段见[参数说明](../PARAMETERS.md)，运行库见[算子库说明](../../../docs/runtime_assets.md)。
