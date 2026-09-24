# Wall-OSS 公开基础配置

`base/` 是唯一配置目录，包含 FP16、W8A16、W4A16、group-wise W4 四份 `[example/input/run]` 模板。原 `public_0.5/` 和 `base/` 都面向同一个公开 Wall-OSS-0.5 底座；此前两套示例格式造成重复，现已合并为这一套。

```sh
python examples/run_model.py --config examples/configs/wall_oss/base/w8a16.toml --check-config
python examples/run_model.py --config examples/configs/wall_oss/base/w8a16.toml
```

使用公开基础 checkpoint、对应量化产物和原浮点源；`fp16_registry_alias` 必须匹配量化来源。模板保留三张相机图像，其顺序/数量与 `camera_names` 一致。公开 `WallOssPolicy.from_checkpoint(...).to("rpu")` 负责构造，`infer` 检查实际图像、状态、mask、指令及 checkpoint 的输入范围。

模板不固定历史请求的前缀范围。需要预构建有限 Graph 范围时，可按实际输入添加 `input.prefix_length_range = [min_rows, max_rows]`；它表示包含图像和指令的实际前缀长度，由 policy 校验，不能套用其他数据的范围。示意图片不代表历史性能或任务质量输入。

原 `public_0.5/fp16.toml`、`public_0.5/w8a16.toml` 分别合并到 `base/fp16.toml`、`base/w8a16.toml`；原 `public_0.5/w4a16_vision_w8.toml` 合并到 `base/w4a16.toml`。分组 INT4 仍单独使用 `base/w4a16_pgrp.toml`。

应用专用 checkpoint、私有数据、实时控制定制和部署特化不由这些模板提供；更换权重名称不会扩大运行时准入。
