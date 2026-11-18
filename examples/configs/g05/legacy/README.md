# G0.5 Legacy 配置

[模型页](../README.md) · [模型配置表](../README.md#configurations) · [权重准备](../../WEIGHTS.md)

[FP16 配置](fp16.toml) 使用已有 K1 RTC/FM release 和预处理 batch。需要按相机数量生成输入时，使用模型页中的 `2cam/` 或 `3cam/` 模板。

在 TOML 中填写 `model.checkpoint`、`model.release_root` 和 `input.batch`。batch 必须包含预处理后的 `samples` 和 `pixel_values`，保留原始 RTC prefix。

从仓库根目录先检查配置，再运行：

```bash
python examples/run.py --config examples/configs/g05/legacy/fp16.toml --check-config
python examples/run.py --config examples/configs/g05/legacy/fp16.toml
```

实际图像、文本和动作范围随配套 batch 确定，核心范围和登记状态见[模型配置表](../README.md#configurations)。

`input.batch` 顶层必须直接包含 `samples` 和 `pixel_values`（release `ReplayRunner` 预处理后的结果）。仓库内没有生成这种文件的脚本；`2cam/`、`3cam/` 入口保存的 `input_batch.pt` 带外层包装，不能直接用于本入口。没有现成 batch 时，使用上一级的 `2cam/` 或 `3cam/` 模板。
