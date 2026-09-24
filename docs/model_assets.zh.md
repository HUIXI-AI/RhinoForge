# 模型资产

简体中文 | [English](model_assets.md)

RhinoForge 不分发模型 checkpoint。请从模型 owner 或授权资产渠道获取，接受其独立
条款，并在使用前核对公开 repository 和 revision。

原始 checkpoint 和本地转换产物都应保留来源记录。运行时准入取决于对应配置和 API
要求的 checkpoint 格式、精度与输入范围。仅有来源身份不能证明模型受支持，参见
[示例](model_support.zh.md)。

## 模型来源与精度

记录公开 checkpoint 来源、revision、许可证和实际精度。离线转换时保留转换命令、
转换器版本和量化元数据，确保加载器能区分 AWQ、INT4、NVFP4 和 W8。
模型权重与请求数据由用户提供，不打包到源码或 wheel。

示例按模型归档在 [`examples/configs/`](../examples/configs/)。
入口和配置说明见[示例索引](../examples/README.md)，模型运行范围由当前 API 前验决定。

## 公开上游来源

[入门指南](getting_started.zh.md)提供固定 revision 的 Qwen3-0.6B 下载命令。
[Qwen3](../examples/configs/qwen3/README.md)、
[Qwen3-VL](../examples/configs/qwen3_vl/README.md)和
[Pi0.5](../examples/configs/pi05/README.md)配置指南说明各自示例要求的 checkpoint
格式、转换选项和输入约束。

公开 Wall-OSS 底座来源为
[`x-square-robot/wall-oss-0.5`](https://huggingface.co/x-square-robot/wall-oss-0.5)。
按 [Wall-OSS 示例](../examples/README.md#公开-wall-oss-与-rhinovla)使用其 processor、normalizer
和动作定义。G0.5 的[配置](../examples/configs/g05/base/fp16.toml)记录了来源、
revision、subfolder 和实现。请查阅模型 owner 的访问与许可条款及
[第三方声明](../THIRD_PARTY_NOTICES.md)；RhinoForge 的源码许可证不授予这些权重的使用权。

保留完整 checkpoint，包括配置、tokenizer 和 processor 文件。Pi0.5 的 SigLIP
权重属于 policy checkpoint。本地转换权重必须符合所选配置要求的格式，不继承原始
checkpoint 的任务质量结论。

## 下载方式

对于 Hugging Face checkpoint，选择与目标配置匹配的 repository 和不可变 revision：

```bash
export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
hf download SOURCE_ID \
  --revision REVISION \
  --local-dir "$RPU_MODEL_CACHE/LOCAL_MODEL_DIRECTORY"
```

所选配置或指南固定了来源、revision 和 subfolder 时，应使用其中的值。示例若仅指定
本地 alias，请从模型 owner 获取 checkpoint，并记录实际 repository 和 revision。
Alias 只映射本地缓存路径，不下载权重或固定上游 revision。gated 模型的访问权由
模型 owner 授予。

当前 Hy-Embodied profile 可以直接下载到 registry 路径：

```bash
hf download tencent/Hy-Embodied-0.5-VLA-UMI \
  --revision 3f53d1f8d2bc587c523cfdc9f1041ceee42c2524 \
  --local-dir "$RPU_MODEL_CACHE/Hy-Embodied-0.5-VLA-UMI"
```

## 已有离线转换器

只为明确指定对应输出格式的配置运行 converter：

```bash
# Qwen3 14B 本地转换示例
python -m rpu_backend.quant.convert_qwen3 \
  --src SOURCE_DIR --dst OUTPUT_DIR --quant-lm-head

# Pi0.5 W8A16
python -m rpu_backend.quant.convert_pi05 --src SOURCE_DIR --dst OUTPUT_DIR

# Pi0.5 mixed W4A16-G32-KV8 评估配置（不是纯 W4）
python -m rpu_backend.quant.convert_pi05 \
  --src SOURCE_DIR --dst OUTPUT_DIR --fake-w4 --real-w4

# Wall-OSS W8A16
python -m rpu_backend.quant.convert_wall_oss_w8a16 \
  --src SOURCE_DIR --dst OUTPUT_DIR

# Wall-OSS group-wise W4 评估配置
python -m rpu_backend.quant.convert_wall_oss \
  --src SOURCE_DIR --dst OUTPUT_DIR --group-size 32
```

Converter 创建新 destination 并拒绝已有目录。转换后让匹配 TOML 指向该输出，并保留
source model 与 converter version 以便复现。

## Runtime 资产独立管理

模型资产不能替代外部运行时前置项：

- 匹配的 sanitized Release Rhino Launch 包，包含头文件和动态库；
- 由 `RPU_KERNEL_LIB_PATH` 选择的匹配合并算子资产，以及相邻的 `.kernels` 清单；
- 匹配的板卡 SDK/runtime，按平台安装说明配置。

这些组件均不存放在仓库、source archive 或 Python 包中。请向运行时提供方获取与源码
版本匹配的资产，并核对其提供的校验值。仅有包版本号不能证明兼容性；构建和运行时应
使用一致的 Launch 路径。所需能力、路径配置和算子清单检查见
[外部运行时资产](runtime_assets.zh.md)。
