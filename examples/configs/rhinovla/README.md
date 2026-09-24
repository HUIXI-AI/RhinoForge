# RhinoVLA 版本配置

仅保留 `v1/` 和 `v3/` 版本目录，两者均使用 `[example/input/run]`，通过公开 `RhinoVLAPolicy.from_factory` 调用模型集成方的可信 factory。原 `public/default.toml` 已合并到 `v1/fp16.toml`。

| 配置 | 入口与范围 |
|---|---|
| `v1/fp16.toml` | v1 FP16 caller factory；三组件 chunk 使用 AUTO |
| `v3/fp16.toml` | v3 FP16 caller factory；保留公开 Vision、prefill、action 冷优化控制 |
| `v3/w8a16.toml` | v3 caller factory；action expert 七类投影 W8A16 |
| `v3/full_expert_w8a16.toml` | v3 caller factory；上述投影及 expert 的 AdaRMS 条件投影 W8A16 |

```sh
python examples/run_model.py --config examples/configs/rhinovla/v1/fp16.toml --check-config
python examples/run_model.py --config examples/configs/rhinovla/v3/fp16.toml --check-config
python examples/run_model.py --config examples/configs/rhinovla/v3/w8a16.toml --check-config
python examples/run_model.py --config examples/configs/rhinovla/v3/full_expert_w8a16.toml --check-config
```

运行前安装并审查相应版本的模型集成，把 `input.runtime_factory` 改成它导出的 `module:callable`，指定匹配的 checkpoint、`factory_config` JSON 和 `request` tensor 字典，保持显式 `trust_model_code=true`。v3 模板显式指向 v3 factory 和 checkpoint 占位路径；它没有内置 v3 完整流水线。factory 必须核对模型版本、精度、维度、预处理与 denoise schedule，再构造公开 adapter。

`factory_config` JSON 原样提供模型集成方定义的附加字段；runner 同时绑定从 alias 或 `input.checkpoint` 解析的 `checkpoint`，以及 `input.num_steps` 对应的 `steps`，重复字段冲突时拒绝。`request` 由 `torch.load(..., weights_only=True)` 读取为映射，再传给 `policy.predict(**request)`。例如：

```json
{"model_root": "/path/to/your/reviewed/model/source"}
```

此 JSON 仅展示常见模型源目录，实际字段由所选 factory 定义。factory、权重和请求全部由调用方管理；示例不加载隐含 fixture、golden 或私有 bundle。

W8 模板还显式绑定 `input.expert_w8a16` 与 `input.full_w8a16`，并以同名布尔值传入 factory JSON。重复值必须类型、取值都一致，数字 `0/1` 不视为布尔值。factory 必须把这些值传到公开 expert 构造 API；runner 会检查绑定的实际 expert 权重、scale 和安装模式，拒绝仅改变精度标签。`full_w8a16` 在此是 **expert 构造参数**：两个 W8 模板都没有选择全 Vision/Language/action-IO 量化。详细映射见 [v3 说明](v3/README.md)。未使用量化字段的既有 FP16 factory 保持原调用参数。

`rpu_execution` 的组件配置必须由 factory 在权重安装前消费，并通过公开 `bind_rhinovla_execution_runtime` 绑定实际 language、vision、action 子模块。公开 facade 会核对 factory 声明的能力和有效配置，拒绝静默忽略。v3 优化组合及条件见 [v3 说明](v3/README.md)。配置检查和运行计时都不代表数值或机器人任务通过。
