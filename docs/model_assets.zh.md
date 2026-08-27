# 模型资产

简体中文 | [English](model_assets.md)

RhinoForge 不再分发模型 checkpoint。请从模型 owner 或授权资产渠道获取，接受其独立
条款，并在使用前核对公开 repository 和 revision。

本页区分 profile 正确性与资产 provenance。v1.0.0 清单固定本次源码发布使用的全部公开
上游身份；本地派生 artifact 不属于该清单，也不会改变[模型支持](model_support.zh.md)中的状态。

## 可运行配置的必需记录

每个可运行配置必须同时发布：

- 模型家族和精确配置；
- source identifier，以及模型 owner 提供时的不可变 revision；
- 模型 license 和访问条件；
- 源 checkpoint 或量化 checkpoint 格式；
- 需要转换时的转换命令与 converter 版本；
- 对应 example TOML；
- 兼容的公开 RhinoForge release tag；
- 所需的 Rhino Launch 和合并算子资产 release version。

不要混用不同受支持配置的模型版本或 runtime 资产版本。

v1.0.0 的机器可读上游清单为
[`release/public-models-v1.0.0.json`](../release/public-models-v1.0.0.json)。它只固定
官方公开仓库和不可变 revision，不记录客户资产，也不会把派生量化目录冒充成公开源
checkpoint。

## Release 身份边界

上述 JSON 清单是本仓库所有公开上游接入在 v1.0.0 中的身份契约。每条记录都包含
公开 repository、不可变 revision、访问条件、已声明的许可状态和 checkpoint 格式；
adapter 依赖模型 owner 代码时，还会记录 implementation repository 与 revision。

该清单不会把 Source-only 或 Limited 配置提升为 Supported，不会授予 gated 资产的访问权，
也不会给本地派生 W8/W4 输出建立身份。除非公开 release 明确列出，派生 checkpoint
始终不属于公开身份清单。

兼容性仅由公开 RhinoForge tag 及所需 Rhino Launch、算子资产的 release version
确定，不依赖内部 source commit/tree、receipt 或 runtime linker/source 映射。

## 公开上游来源

上述 JSON 清单是 repository、revision、访问条件、许可声明和 checkpoint 格式的权威
记录。Wall 行将公开 `wall-oss-0.5` checkpoint 与 Apache-2.0 的 `wall-x` 实现绑定，
但 checkpoint 仓库未声明权重许可。G0.5 行绑定公开 Galaxea 实现与 gated 的
`g05-base` 资产；这些资产仍受 G0.5 Community License 的非商业限制。

Pi0.5 使用的 SigLIP 属于固定的 Pi0.5 bundle，不建立独立 RhinoForge checkpoint
身份。除非公开 release 明确列出，W8/W4 派生资产不属于可运行 release 声明。

## 下载方式

Release 行给出 Hugging Face 来源和不可变 revision 时：

```bash
export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
hf download SOURCE_ID \
  --revision REVISION \
  --local-dir "$RPU_MODEL_CACHE/LOCAL_MODEL_DIRECTORY"
```

来源、revision、subfolder（如有）和访问条款必须取自同一 release 行。gated 模型的
访问权由模型 owner 授予。

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

# Pi0.5 packed W4 评估配置
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

模型资产不能替代两个受限运行前置项：

- 公开 RhinoForge tag 要求的 Rhino Launch 二进制开发包 release；
- 由 `RPU_KERNEL_LIB_PATH` 选择的合并算子资产 release，以及它的相邻 kernel
  manifest。

两者均不存放在仓库、source archive 或 Python 包中。兼容性由公开 RhinoForge tag
以及该 tag 文档列出的两项 runtime 资产 release version 确定，不要求内部 commit、
linker 映射或 source-build 身份。请从授权渠道获取并校验随包提供的 checksum；下载、
安装和撤销流程见[受限运行时资产](runtime_assets.zh.md)。
