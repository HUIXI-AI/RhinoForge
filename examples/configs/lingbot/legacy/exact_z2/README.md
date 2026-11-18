# LingBot v2 历史 exact-Z2 配置

<a id="lingbot-legacy-配置"></a>

[模型页](../../README.md) · [模型配置表](../../README.md#configurations) · [权重准备](../../../WEIGHTS.md)

[6B W8A16 配置](6b/w8a16.toml) 保留历史掩码和固定 P225 执行约束，用于复现 exact-Z2 历史记录。常规 FP16/W8A16 输入请使用 [v2 配置](../../v2/README.md)。

准备 `lingbot-vla-v2-6b` 权重和 `input.observation` 指定的 exact-Z2 输入。模板记录的 `prefix_length_range` 为 215–225，并保留该路径所需的 opt-in；这些条件不适用于其他 observation。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/lingbot/legacy/exact_z2/6b/w8a16.toml --check-config
python examples/run.py --config examples/configs/lingbot/legacy/exact_z2/6b/w8a16.toml
```

配置检查只确认语法和 profile。此配置只覆盖上述固定输入，结论不能推广到普通 v2 输入。

`input.observation` 是复现历史 exact-Z2 记录用的固定输入（三路 256×256 图像、真实前缀长度在 215–225 之间）。仓库内没有生成这种文件的脚本，也没有随仓库发布该文件；没有原始文件时请使用 [v2 配置](../../v2/README.md)。
