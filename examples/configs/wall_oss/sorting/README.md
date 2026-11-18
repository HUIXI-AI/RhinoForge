# Wall-OSS Sorting 配置

[模型页](../README.md) · [模型配置表](../README.md#configurations) · [权重准备](../../WEIGHTS.md)

Sorting 使用分拣任务的配套三路相机 batch。根据已准备的权重选择配方；量化配置仍需填写 FP16 checkpoint。

| 配方 | 额外权重字段 | 配置 |
|---|---|---|
| FP16 | 无 | [fp16.toml](fp16.toml) |
| W8A16 | `model.w8_checkpoint` | [w8a16.toml](w8a16.toml) |
| W8A16 + Action W4A16 G32 | `model.w8_checkpoint`、`model.action_w4_checkpoint` | [w8a16_action_w4a16_g32.toml](w8a16_action_w4a16_g32.toml) |

填写 `model.fp16_checkpoint`、`input.batch` 及所选配方的额外路径，设置 `input.inference_delay`。从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/wall_oss/sorting/w8a16.toml --check-config
python examples/run.py --config examples/configs/wall_oss/sorting/w8a16.toml
```

此入口不接受 `--hwperf` 或 `--profile`。配置检查仅检查语法；输入规格、动作范围和核心范围以配套 batch 及[模型配置表](../README.md#configurations)为准，Sorting 不继承 Base 的登记 profile。

<a id="input-files"></a>

## 输入文件准备

`input.batch` 是 `torch.save` 保存的 CPU 字典，读取时使用 `weights_only=True`：

| 键 | 形状与类型 | 说明 |
|---|---|---|
| `images` | uint8 `[3,252,448,3]` | face_view、left_wrist_view、right_wrist_view，RGB；入口不缩放图像 |
| `instruction` | 非空字符串 | 本次分拣指令 |
| `state` | 浮点 `[26]` | 物理模型空间的 proprioception |
| `state_mask` / `dof_mask` | bool `[26]` | 前 20 维为 true，后 6 维为 false |
| `noise` | 浮点 `[1,20,26]` | 固定初始噪声 |
| `prefix_actions` | 浮点 `[20,26]` | 仅 `inference_delay` > 0 时提供；为 0 时不能出现 |

仓库内没有生成脚本。以下代码把同一观测的三张图、状态和指令整理为一个文件：

```python
from pathlib import Path

import numpy as np
import torch
from PIL import Image

paths = ["/path/to/face.png", "/path/to/left_wrist.png", "/path/to/right_wrist.png"]
images = torch.stack([torch.from_numpy(np.array(Image.open(p).convert("RGB"), copy=True)) for p in paths])
assert images.shape == (3, 252, 448, 3)
state = torch.load("/path/to/state.pt", map_location="cpu", weights_only=True)
assert state.shape == (26,)
active = torch.tensor([True] * 20 + [False] * 6)
batch = {
    "images": images,
    "instruction": Path("/path/to/instruction.txt").read_text().strip(),
    "state": state.float(),
    "state_mask": active,
    "noise": torch.randn(1, 20, 26, generator=torch.Generator().manual_seed(0)),
    "dof_mask": active.clone(),
}
# inference_delay > 0 时再加入：batch["prefix_actions"] = <前次物理动作，形状 [20,26]>
torch.save(batch, "/path/to/wrc_v728_batch.pt")
```

batch 在生成 `report.json` 之前读取，格式不符时只在终端报错。
