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

`input.observation` 是复现历史 exact-Z2 记录用的固定输入。仓库内没有生成这种文件的脚本，也没有随仓库发布该文件。

- **文件契约**：`torch.save` 的 tensor 字典，以 `weights_only=True` 加载，键名不做改写。必需键 `images` float `[3,256,1536]`、`lang_tokens` int64 `[1,L]`、`state`、`noise`；可选 `img_masks` bool `[3]`（须全为 true）与 `lang_masks` bool `[1,L]`。
- **长度**：真实前缀为 3×66+16+T 行，须在 215–225 之间，所以有效指令 token 数 T 只能是 1–11，相机必须是 3 路 256×256。
- **原始输入**：也可以删除 `observation` 行，改填 `images`（3 张）、`instruction`、`proprioception`，noise 由 `input.seed` 生成；这种写法没有对应的 CPU golden。
- **检查时机**：`--check-config` 不检查这些字段，不符合时在运行时报错。

其他相机数或指令长度（如 2 路或 T64）超出此配置范围，请使用 [v2 配置](../../v2/README.md)。
