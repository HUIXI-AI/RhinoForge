# 示例与运行范围

简体中文 | [English](model_support.md)

[`examples/`](../examples/README.md) 提供统一的 `run_model.py` 入口和模型专用脚本。
TOML 配置在 [`examples/configs/`](../examples/configs/README.md) 下按模型、组件、
精度和工作量归档。

选择配置并准备对应 checkpoint 与输入后，先运行
`python examples/run_model.py --config PATH --check-config`。该命令不加载权重或访问
板卡，只检查配置，不验证 checkpoint 或认证模型质量。对应运行合同见
[模型运行](model_testing.zh.md)、[API 参考](api_reference.zh.md)和
[量化说明](quantization.zh.md)。

## 通用运行边界

每个 checkpoint 和请求的实际范围由运行时准入检查决定。配置文件不代表其他尺寸、
精度、核数、图像数量或输入长度也适用。Source-only 和受控示例保留各自精确的输入
限制与 opt-in 开关；历史验证记录不代表当前运行时，也不构成新的模型认证。

模型权重、请求数据、Rhino Launch 和算子资产分别获取，其格式、来源和许可要求见
[模型资产](model_assets.zh.md)与[运行时资产](runtime_assets.zh.md)。保留公开
Wall-OSS 底座适配；公开交付不包含私有应用模型、数据或部署定制。
