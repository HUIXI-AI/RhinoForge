# Wall-OSS Base 配置

[模型页](../README.md) · [模型配置表](../README.md#configurations) · [核数支持矩阵](../../../../docs/planner_core_topology.zh.md) · [权重准备](../../WEIGHTS.md)

先匹配 batch 的相机数量与图像尺寸，再选择 FP16、W8A16 或 W8A16 + Action NVFP4。

| 相机数量 | 224×224 | 256×256 |
|---|---|---|
| 2 路 | [2cam/224/](2cam/224/) | [2cam/256/](2cam/256/) |
| 3 路 | [3cam/224/](3cam/224/) | [3cam/256/](3cam/256/) |

## 准备与运行

填写 `model.fp16_checkpoint` 和 `input.batch_template`；W8 配方还需 `model.w8_checkpoint`。batch 路径中的 `{cameras}`、`{image_size}`、`{text_tokens}` 会按输入参数展开。模板默认使用 32 个指令 token、32 步动作输出和 10 步去噪，修改输入规格时需匹配实际 batch。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/wall_oss/base/2cam/224/fp16.toml --check-config
python examples/run.py --config examples/configs/wall_oss/base/2cam/224/fp16.toml
```

配置检查允许占位路径。Base 入口用于延迟测量，不接受 `--hwperf` 或 `--profile`；核心布局和状态见[模型配置表](../README.md#configurations)。

<a id="text-limit"></a>

## 文本长度上限

8 核 prefill 的真实 prefix 不能超过 640 行。prefix 由图像 token、固定提示和指令组成（instruction 范围）：

| 几何 | 真实 prefix | T 上限 |
|---|---|---|
| 2cam 224 | 244+T | 396 |
| 2cam 256 | 278+T | 362 |
| 3cam 224 | 319+T | 321 |
| 3cam 256 | 370+T | 270 |

planner 按真实 prefix 选择执行行数：≤384 行补到下一个 16 的倍数，单 chunk 执行；385–448、449–512、513–640 行分别执行 448、512、640 行，等分为两个 chunk（224、256、320）。超过 640 行时，入口在加载权重后报错 `equal-two prefill has no safe two-chunk execution profile`。原因是 672–768 行的等分 chunk（336–384）都通不过 SDPA chunk 校验。

`text_tokens` 按指令单独编码计数，真实 prefix 按指令在提示中的实际切分计数，两者偶尔相差 1 行（例如单词指令 `Grab`）。真实值记录在 `report.json` 的 `input.real_prefix_len`。

这些边界来自 2026-10-23 板测：T1–256 各几何都通过；每种几何 prefix 正好 640 行的点通过，641 行的点报错。结果见性能报告。


<a id="input-files"></a>

## 输入文件准备

`input.batch_template` 展开后的每个文件是 `torch.save` 保存的字典，键必须恰好为以下六个，全部为 CPU tensor（`instruction` 除外）：

| 键 | 形状与类型 | 说明 |
|---|---|---|
| `images` | uint8 `[cameras, H, W, 3]`，H=W=`image_size` | RGB，相机顺序 face_view、left_wrist_view、right_wrist_view（2cam 取前两路） |
| `instruction` | 非空字符串 | token 数必须等于 `text_tokens`；入口不截断也不补齐，该检查在权重加载之后执行 |
| `proprioception` | fp16 / fp32 `[26]` | 机器人状态 |
| `state_mask` | bool `[26]` | 必须是前 7 维为 true |
| `noise` | fp16 / fp32 `[1,32,26]` | 去噪初始噪声 |
| `dof_mask` | fp16 / fp32 `[1,32,26]` | 前 7 维为 1.0，其余为 0 |

仓库内没有生成脚本；可参考测试夹具 `tests/test_wall_base_benchmark_inputs.py` 中的 `observation()` 构造。格式不符时错误写入 `report.json` 的 `error`，进程返回 1。
