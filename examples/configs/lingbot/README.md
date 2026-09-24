# LingBot-VLA v2

`v2/6b_fp16.toml` 和 `v2/6b_w8a16.toml` 使用同一套 `[example/input/run]` 格式，分别选择不同的实际精度。默认入口使用最新 W8A16 配置；FP16 保留普通公开 policy 的评估入口。v1 配置已移除。

```sh
python examples/run_model.py --config examples/configs/lingbot/v2/6b_w8a16.toml --check-config
python examples/run_model.py --config examples/configs/lingbot/v2/6b_fp16.toml --check-config
```

两个配置通过 `Lingbot2Policy.from_checkpoint` 从调用者提供的同一浮点源 checkpoint 构造；W8A16 在装载时选择量化，文件名与 `dtype` 一致。FP16 使用三张原始图像、指令和状态，Graph 前缀范围由 policy 根据实际输入计算。W8A16 使用显式的 tensor observation 和 `[215, 225]` 前缀范围，保留其 exact-Z2 选择。

示意图片可替换为调用者图像；预处理尺寸、相机数、状态和 checkpoint 均由公开 policy 检查。`ALLOW_UNVALIDATED` 是显式评估许可，配置文件或性能计时不代表数值或机器人任务通过。

旧 `lingbot2.toml` 已移至 `v2/6b_w8a16.toml`；旧 `v2_6b/fp16.toml` 已合并为 `v2/6b_fp16.toml`。目录中不再同时保留两套示例格式。
