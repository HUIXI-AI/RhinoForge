# 升级到 v1.2.0

简体中文 | [English](upgrading_1_2.md)

当前正式版为 `1.2.0`，主要变化见[发行说明](../CHANGELOG.md)。
发行包名为 `rhinoforge`，Python 使用 `rpu_backend` 和 `torch.rpu`。

## 安装

1. 按[入门指南](getting_started.zh.md)创建新的 Python 3.12 Conda 环境。
   [pyproject.toml](../pyproject.toml)固定 PyTorch 2.10.0 和 Transformers 5.3.0；
   按模型选择 `vision`、`vla` 或 `audio` 可选依赖。
2. 从[资产申请入口](runtime_assets.zh.md#申请运行时资产)获取
   `RhinoForge-runtime-v1.2.0.tar.gz` 及校验和。核对归档，保留 main、DDR-expansion
   两份资产及各自清单。Rhino Launch 同样使用 `1.2.0`；组件修订见
   [依赖清单](../release/runtime-dependencies.json)。
3. 更新 `RHINO_LAUNCH_ROOT`、`RPU_KERNEL_LIB_PATH` 和
   `RPU_KERNEL_LIB_PATH_EXPANSION`，重新构建扩展并启动新进程。
   旧环境与旧资产配套保留，供回退使用。
4. 运行 `python -m pip check`，按下方命令检查所选 TOML；准备配套权重和输入后，
   在空闲设备上运行一次。

## 选择配置

从[模型目录](../examples/configs/README.md)选择当前模板，复制到本地文件，
按模型指南填写资产和输入路径：

```bash
cp examples/configs/qwen3/text/0_6b/fp16.toml qwen3.local.toml
python -m examples.run --config qwen3.local.toml --check-config
python -m examples.run --config qwen3.local.toml
```

配置名标明模型、组件、精度和工作负载，详见[命名规则](../examples/configs/NAMING.md)。

| 模型 | 输入要求 |
|---|---|
| [GR00T](../examples/configs/gr00t/README.md) | 匹配 checkpoint 的相机和状态观测张量；设置 `input.observation`、`input.seed`、`input.num_steps`。 |
| [InternVLA / NavDP](../examples/configs/internvla/README.md) | 完整链 `input.fixture`、`input.asset_manifest`，以及 `input.sample_num`、`input.seed`。 |
| [Pi0.5](../examples/configs/pi05/README.md) | 完整策略 batch 与配套 checkpoint；独立 SigLIP 调用使用组件 adapter。 |
| [LingBot v2](../examples/configs/lingbot/v2/README.md) | 选择相机数量和语言长度，提供 checkpoint、训练 YAML 及配套 case、observation 或图像/状态输入；保持相机顺序与动作归一化一致。 |
| [Qwen3.5 MoE](../examples/configs/qwen3_5/README.md) | 长文本模板使用 `prompt_file`、`prompt_sha256`、`input_ids_sha256` 和明确的 prefill/decode 长度；采用模板的 `run.timing_mode`。 |
| [RhinoVLA](../examples/configs/rhinovla/README.md) | v1 使用固定源码、checkpoint 和 fixture；v3 按指南配置 runtime factory 和配对的预处理输入。 |

模板要求 `input_ids_sha256` 时，用配套 tokenizer 设置 `add_special_tokens=False`，
对前 `prefill_tokens` 个连续 INT64 token ID 的字节计算 SHA256。文本须包含足够 token，
该序列不能含 EOS/PAD。构造过程见[输入实现](../examples/common/generation.py)。

## 日志与转换

需要生成文本和普通进度时，在 TOML 顶层添加：

```toml
[logging]
level = "INFO"
summary = true
```

默认日志级别为 `ERROR`，`summary` 独立控制模型和运行摘要。
输出文件与字段见[日志与结果](../examples/RESULTS.md)。

Wall-OSS W4 转换默认 `--group-size 32`，两个 decoder expert 的投影均使用 INT4，
包含 expert-0 Down。逐通道 scale 使用 `--group-size 0`；非空 `--keep-int8` 会被拒绝。
按[转换命令](../examples/configs/WEIGHTS.md#cmd-wall)生成新目录。
分拣配方只从 W4 目录加载动作专家。

更换运行时资产、模型精度或 Graph 冷配置后，使用新进程。权重转换失败或只完成一部分时，
重新加载干净的原始权重。参数生效时机见[运行时配置](runtime_config.zh.md)。
