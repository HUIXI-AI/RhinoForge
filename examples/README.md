# 模型示例

<a id="model-examples"></a>
<a id="choose-a-model"></a>

从[模型目录](configs/README.md)选择模型，再按模型页的三个部分操作：**配置说明 → 准备与运行 → 结果说明**。配置表同时给出 TOML 入口和对应的权重获取步骤。

| 要做什么 | 查看哪里 |
|---|---|
| 选择模型、版本和配置 | [模型目录](configs/README.md)、[全部 TOML 索引](configs/CONFIGURATIONS.md) |
| 理解权重来源、量化与加载的关系 | [权重准备](configs/WEIGHTS.md)；具体下载和转换命令在模型页 |
| 填写输入、线程、核数和采样参数 | 模型页“准备与运行”、[通用参数](configs/PARAMETERS.md) |
| 理解文件名和定位原记录中的配置 | [命名规则](configs/NAMING.md)、[路径对照](configs/config_names.json) |
| 查看日志与输出 | 模型页“结果说明”、[日志与结果](RESULTS.md) |
| 查看登记范围和运行库要求 | [支持矩阵](../docs/model_support.zh.md)、[运行库准备](../docs/runtime_assets.md) |
| 运行 LightNav-0 缓存评测 | [LightNav-0 R1 指南](../docs/lightnav_0_r1.zh.md) |

<a id="run"></a>

## 运行一个配置

在已安装后端和模型依赖的环境中，从仓库根目录执行。先按模型页准备权重和输入，再复制所选模板：

```bash
cp examples/configs/g05/3cam/fp16.toml /tmp/g05.toml
# 按模型页填写配套权重、输入和 run.output_dir。
vi /tmp/g05.toml
python examples/run.py --config /tmp/g05.toml --check-config
python examples/run.py --config /tmp/g05.toml
```

`--check-config` 检查配置结构及登记关系，无需加载 Torch/设备。最后一条命令执行模型加载和推理。RhinoVLA 的 `controlled/` 配置使用模型页给出的独立验证入口。

<a id="logging-and-output"></a>
<a id="日志与输出"></a>
<a id="logging-and-performance-output"></a>
<a id="日志与性能输出"></a>
<a id="日志等级对照"></a>
<a id="模型与性能摘要"></a>
<a id="results"></a>

## 查看输出

默认终端显示 `[MODEL]` 配置摘要、`[PERF]` 本次运行摘要和错误信息。需要加载过程或生成文本时，在 TOML 中设置 `[logging].level="INFO"`。完整字段、日志文件和结果解读见[日志与结果](RESULTS.md)。

<a id="已归档示例"></a>

LingBot [exact-Z2](configs/lingbot/legacy/exact_z2/README.md)、各模型的 `legacy/` 和 RhinoVLA 独立验证配置，使用各自页面列出的资产和输入。它们也列在[完整配置索引](configs/CONFIGURATIONS.md)中。

版本升级见[升级指南](../docs/upgrading_1_2.zh.md)；语音 API 见[Qwen3 ASR/TTS](../docs/qwen3_audio.zh.md)。
