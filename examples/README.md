# 模型示例

从[模型目录](configs/README.md)选择模型和 TOML，准备对应权重与输入，再运行 `examples/run.py`。命令从仓库根目录执行；环境需已安装后端、模型依赖及[运行库](../docs/runtime_assets.zh.md)。

本页说明配置与运行，不列性能或数值误差测量结果；维护位置见文档分工。

<a id="model-examples"></a>
<a id="choose-a-model"></a>

| 需要了解的内容 | 查看位置 |
|---|---|
| 模型、变体和配置入口 | [配置总览](configs/README.md) |
| 4/6/8 核支持与验证状态 | [核数支持矩阵](../docs/planner_core_topology.zh.md) |
| 权重目录和准备步骤 | [权重准备](configs/WEIGHTS.md) |
| 输入、运行和输出参数 | [通用参数](configs/PARAMETERS.md)、对应模型页 |
| 文件名和路径规则 | [命名规则](configs/NAMING.md) |

<a id="run"></a>
<a id="运行一个配置"></a>

## 准备并运行

1. 在模型页确认任务、输入规格、精度配方和 4/6/8 核支持情况。每个模型页的“全部配置与输入范围”列出该模型的全部 TOML 路径。
2. 复制所选 TOML，按[权重准备](configs/WEIGHTS.md)定位资产，填写输入路径和输出目录。`registry_alias` 指向本地缓存中的目录，不会自动下载权重。
3. 先检查配置，再运行模型。以下以 Qwen3 4B FP16 为例：

```bash
python examples/run.py --config examples/configs/qwen3/text/4b/fp16.toml --check-config
python examples/run.py --config examples/configs/qwen3/text/4b/fp16.toml
```

`--check-config` 成功时显示 `configuration OK`，表示字段和登记关系通过检查；权重文件、输入张量与设备环境还需在实际运行时检查。输入、权重、核数或累加选项发生变化后，重新检查并启动新进程。

换核数或改个别字段时可以不复制 TOML：`--recipe w8a16_text_head` 选同一模型目录下的另一个配方，`--cores 6` 覆盖核数，`--input p1024_d128`（图文如 `i896_p1024_d32`，VLA 如 `3cam_t64`，ASR 如 `6s`）套用一组[输入预设](configs/PARAMETERS.md#input-presets)，`--set <表>.<键>=<值>` 覆盖一个输入、运行或执行字段，`--emit-toml <路径>` 把覆盖后的配置保存成可直接运行的 TOML。可改范围见[命令行覆盖](configs/PARAMETERS.md#cli-overrides)。

<a id="results"></a>
<a id="logging-and-performance-output"></a>

## 查看结果

终端默认显示模型与性能摘要，结果文件写入 `run.output_dir`。先核对实际输入、输出形状及运行次数，再比较耗时；日志设置和计时字段见[结果说明](RESULTS.md)。

<a id="已归档示例"></a>

受控配置使用各自页面列出的资产和输入；其中 RhinoVLA `controlled/` 是带固定开关组合的 v3 源码入口。归档目录用于复现历史配置，支持范围以各模型页配置表中的支持状态为准。
