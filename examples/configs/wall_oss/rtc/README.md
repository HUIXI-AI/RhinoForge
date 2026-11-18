# Wall-OSS RTC 配置

[模型页](../README.md) · [模型配置表](../README.md#configurations) · [权重准备](../../WEIGHTS.md)

RTC 使用配套 checkpoint、三路图像、文本指令、机器人状态和 `inference_delay`。提供 [FP16](fp16.toml) 与 [W8A16](w8a16.toml) 两个模板。

## 准备与运行

填写 `model.checkpoint` 和图像路径，按实际请求设置 `instruction`、`proprioception`、`inference_delay`。W8A16 模板还需 `model.fp16_checkpoint`。模板 prefix 长度范围为 599–640，推理延迟默认为 0。

`model.registry_alias` 只用于匹配 profile，须保留且与 profile 一致，不决定读哪份权重：权重总是取 `model.checkpoint`（W8A16 的 FP16 源取 `model.fp16_checkpoint`）。不要删除 `checkpoint`：删除后改读缓存中的上游 `wall-oss-0.5`，它不是 v728 权重，加载时的 `config.yml` 检查会拒绝。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/wall_oss/rtc/fp16.toml --check-config
python examples/run.py --config examples/configs/wall_oss/rtc/fp16.toml
```

模板的 `[model.opt_in]` 为空，不需要填写；按[模型配置表](../README.md#configurations)确认状态与核数范围。配置检查通过或完成一次动作推理，都不能替代 RTC 任务质量验证。

此入口不需要预先准备 tensor 文件：三路图片（face、left_wrist、right_wrist）填普通图片路径，`instruction`、`proprioception` 和 `inference_delay` 直接写在 TOML 中。图片不存在时在运行时报 `FileNotFoundError`。

`inference_delay` 取 0–10，省略为 0。非零时还要写 `input.rtc_prefix_actions`：上一段已执行的物理动作，`[L,26]` 的二维数组，`inference_delay ≤ L ≤ 20`，入口只使用前 `inference_delay` 行；为 0 时不能写。例如 `inference_delay = 3` 时写 3 行、每行 26 个数。
