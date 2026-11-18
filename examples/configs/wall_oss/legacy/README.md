# Wall-OSS Legacy 配置

[模型页](../README.md) · [模型配置表](../README.md#configurations) · [权重准备](../../WEIGHTS.md)

此入口直接接收三路图像、文本指令、机器人状态及状态掩码。按已有权重选择精度：

| 精度 | 配置 |
|---|---|
| FP16 | [fp16.toml](fp16.toml) |
| W8A16 | [w8a16.toml](w8a16.toml) |
| W4A16 per-channel | [w4a16_channel.toml](w4a16_channel.toml) |
| W4A16 G32 | [w4a16_g32.toml](w4a16_g32.toml) |

填写 `input.images`、`instruction`、`proprioception` 和 `state_mask`，保持相机顺序与 `camera_names` 一致。模板记录 prefix 长度范围 545–640；量化模板还引用 FP16 权重别名 `fp16_registry_alias`。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/wall_oss/legacy/fp16.toml --check-config
python examples/run.py --config examples/configs/wall_oss/legacy/fp16.toml
```

输入图像尺寸、动作输出和核心范围见[模型配置表](../README.md#configurations)。配置检查不会加载图片或验证机器人状态。

此入口不需要预先准备 tensor 文件：`input.images` 填普通图片路径（PIL 可读，按 RGB 读取，数量与 `camera_names` 一致），`instruction`、`proprioception`（26 个数）和 `state_mask`（26 个 0/1）直接写在 TOML 中。图片不存在时在运行时报 `FileNotFoundError`。
