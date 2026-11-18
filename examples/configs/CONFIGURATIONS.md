# 公开配置索引

由 `python scripts/check_examples.py --write` 生成，列出公开示例 TOML。登记状态来自各配置的 profile。配置差异、权重来源和路径绑定见模型指南。

[选型与使用](README.md) · [登记支持矩阵](../../docs/model_matrix.zh.md) · [权重准备](WEIGHTS.md) · [通用参数](PARAMETERS.md)

CPU 列为请求值，`继承` 表示按运行器或环境设置。固定 P/D 模板指定输入长度、生成长度及计时方式。

## dinov3

[配置说明](dinov3/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [dinov3/vit_b/fp16.toml](dinov3/vit_b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16` | 继承 | `SUPPORTED`<br>`DINOV3_VIT_B_SUPPORTED` |

## gemma4

[配置说明](gemma4/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [gemma4/e4b/fp16.toml](gemma4/e4b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |

## gr00t

[配置说明](gr00t/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [gr00t/n1_7_droid/224/2cam/fp16.toml](gr00t/n1_7_droid/224/2cam/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；图像=224；T=128 | 8 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [gr00t/n1_7_droid/224/2cam/w8a16.toml](gr00t/n1_7_droid/224/2cam/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；图像=224；T=128 | 8 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [gr00t/n1_7_droid/224/fp16.toml](gr00t/n1_7_droid/224/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；图像=224；T=32 | 8 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [gr00t/n1_7_droid/224/w8a16.toml](gr00t/n1_7_droid/224/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；图像=224；T=32 | 8 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [gr00t/n1_7_droid/256/2cam/fp16.toml](gr00t/n1_7_droid/256/2cam/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；图像=256；T=128 | 8 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [gr00t/n1_7_droid/256/2cam/w8a16.toml](gr00t/n1_7_droid/256/2cam/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；图像=256；T=128 | 8 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [gr00t/n1_7_droid/256/fp16.toml](gr00t/n1_7_droid/256/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；图像=256；T=32 | 8 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [gr00t/n1_7_droid/256/w8a16.toml](gr00t/n1_7_droid/256/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；图像=256；T=32 | 8 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |

## halo

[配置说明](halo/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [halo/action_expert/fp16.toml](halo/action_expert/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16` | 继承 | `SUPPORTED_LIMITED`<br>`HALO_SINGLE_ACTION_EXPERT_STEP_SUPPORTED` |

## hy_embodied

[配置说明](hy_embodied/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [hy_embodied/umi/fp16.toml](hy_embodied/umi/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16` | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |

## hy_vla

[配置说明](hy_vla/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [hy_vla/2cam/fp16.toml](hy_vla/2cam/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；T=32；TP=8 | 12（运行器默认） | 未登记；模型专用模板 |
| [hy_vla/2cam/w8a16.toml](hy_vla/2cam/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；T=32；TP=8 | 12（运行器默认） | 未登记；模型专用模板 |
| [hy_vla/2cam/w8a16_action_nvfp4.toml](hy_vla/2cam/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_nvfp4`；相机=2；T=32；TP=8 | 12（运行器默认） | 未登记；模型专用模板 |
| [hy_vla/3cam/fp16.toml](hy_vla/3cam/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；T=32；TP=8 | 12（运行器默认） | 未登记；模型专用模板 |
| [hy_vla/3cam/w8a16.toml](hy_vla/3cam/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；T=32；TP=8 | 12（运行器默认） | 未登记；模型专用模板 |
| [hy_vla/3cam/w8a16_action_nvfp4.toml](hy_vla/3cam/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_nvfp4`；相机=3；T=32；TP=8 | 12（运行器默认） | 未登记；模型专用模板 |

## internvla

[配置说明](internvla/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [internvla/navdp/w8a16_system2.toml](internvla/navdp/w8a16_system2.toml) | [run.py](../run.py)<br>运行模板 | `fp16` | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |

## lingbot

[配置说明](lingbot/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [lingbot/legacy/exact_z2/6b/w8a16.toml](lingbot/legacy/exact_z2/6b/w8a16.toml) | [run.py](../run.py)<br>历史复现 | `w8a16` | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [lingbot/v1/4b/fp16_cpu_vision.toml](lingbot/v1/4b/fp16_cpu_vision.toml) | [run.py](../run.py)<br>运行模板 | `cpu-fp32+rpu-fp16` | 继承 | `SUPPORTED_LIMITED`<br>`LINGBOT_V1_SHIPPED_MIXED_VISION_PATH` |
| [lingbot/v2/2cam/fp16.toml](lingbot/v2/2cam/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；有效T=32；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/2cam/w8a16.toml](lingbot/v2/2cam/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；有效T=32；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/3cam/fp16.toml](lingbot/v2/3cam/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；有效T=32；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/3cam/w8a16.toml](lingbot/v2/3cam/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；有效T=32；TP=8 | 1 | 未登记；模型专用模板 |

## llama

[配置说明](llama/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [llama/3_2_1b/fp16.toml](llama/3_2_1b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED`<br>`LLAMA_3_2_1B_SUPPORTED` |

## pi05

[配置说明](pi05/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [pi05/2cam/h10/fp16.toml](pi05/2cam/h10/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；T=32；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/2cam/h10/w8a16.toml](pi05/2cam/h10/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；T=32；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/2cam/h10/w8a16_action_nvfp4.toml](pi05/2cam/h10/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_action_nvfp4`；相机=2；T=32 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/2cam/h10/w8a16_prefill_w8a8_action_nvfp4.toml](pi05/2cam/h10/w8a16_prefill_w8a8_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_prefill_w8a8_action_nvfp4`；相机=2；T=32 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/2cam/h50/fp16.toml](pi05/2cam/h50/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；T=32；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/2cam/h50/w8a16.toml](pi05/2cam/h50/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；T=32；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/2cam/h50/w8a16_action_nvfp4.toml](pi05/2cam/h50/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_action_nvfp4`；相机=2；T=32 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/2cam/h50/w8a16_prefill_w8a8_action_nvfp4.toml](pi05/2cam/h50/w8a16_prefill_w8a8_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_prefill_w8a8_action_nvfp4`；相机=2；T=32 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/3cam/h10/fp16.toml](pi05/3cam/h10/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；T=32；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/3cam/h10/w8a16.toml](pi05/3cam/h10/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；T=32；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/3cam/h10/w8a16_action_nvfp4.toml](pi05/3cam/h10/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_action_nvfp4`；相机=3；T=32 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/3cam/h10/w8a16_prefill_w8a8_action_nvfp4.toml](pi05/3cam/h10/w8a16_prefill_w8a8_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_prefill_w8a8_action_nvfp4`；相机=3；T=32 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/3cam/h50/fp16.toml](pi05/3cam/h50/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；T=32；TP=8 | 8（运行器默认） | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [pi05/3cam/h50/w8a16.toml](pi05/3cam/h50/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；T=32；TP=8 | 8（运行器默认） | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [pi05/3cam/h50/w8a16_action_nvfp4.toml](pi05/3cam/h50/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_action_nvfp4`；相机=3；T=32 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/3cam/h50/w8a16_prefill_w8a8_action_nvfp4.toml](pi05/3cam/h50/w8a16_prefill_w8a8_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_prefill_w8a8_action_nvfp4`；相机=3；T=32 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/dual_arm/fp16.toml](pi05/kuavo/dual_arm/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；原帧=[848, 480]；T=160；动作维=16；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/dual_arm/t96/fp16.toml](pi05/kuavo/dual_arm/t96/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；原帧=[848, 480]；T=96；动作维=16；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/dual_arm/t96/w8a16.toml](pi05/kuavo/dual_arm/t96/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；原帧=[848, 480]；T=96；动作维=16；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/dual_arm/w8a16.toml](pi05/kuavo/dual_arm/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；原帧=[848, 480]；T=160；动作维=16；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/sim/dual_arm/fp16.toml](pi05/kuavo/sim/dual_arm/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；原帧=[640, 480]；T=160；动作维=16；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/sim/dual_arm/t96/fp16.toml](pi05/kuavo/sim/dual_arm/t96/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；原帧=[640, 480]；T=96；动作维=16；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/sim/dual_arm/t96/w8a16.toml](pi05/kuavo/sim/dual_arm/t96/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；原帧=[640, 480]；T=96；动作维=16；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/sim/dual_arm/w8a16.toml](pi05/kuavo/sim/dual_arm/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；原帧=[640, 480]；T=160；动作维=16；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/sim/single_arm/fp16.toml](pi05/kuavo/sim/single_arm/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；原帧=[640, 480]；T=128；动作维=8；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/sim/single_arm/t64/fp16.toml](pi05/kuavo/sim/single_arm/t64/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；原帧=[640, 480]；T=64；动作维=8；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/sim/single_arm/t64/w8a16.toml](pi05/kuavo/sim/single_arm/t64/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；原帧=[640, 480]；T=64；动作维=8；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/sim/single_arm/w8a16.toml](pi05/kuavo/sim/single_arm/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；原帧=[640, 480]；T=128；动作维=8；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/sim/wholebody/fp16.toml](pi05/kuavo/sim/wholebody/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；原帧=[640, 480]；T=192；动作维=20；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/sim/wholebody/w8a16.toml](pi05/kuavo/sim/wholebody/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；原帧=[640, 480]；T=192；动作维=20；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/single_arm/fp16.toml](pi05/kuavo/single_arm/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；原帧=[848, 480]；T=128；动作维=8；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/single_arm/t64/fp16.toml](pi05/kuavo/single_arm/t64/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；原帧=[848, 480]；T=64；动作维=8；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/single_arm/t64/w8a16.toml](pi05/kuavo/single_arm/t64/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；原帧=[848, 480]；T=64；动作维=8；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/single_arm/w8a16.toml](pi05/kuavo/single_arm/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；原帧=[848, 480]；T=128；动作维=8；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/wholebody/fp16.toml](pi05/kuavo/wholebody/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；原帧=[848, 480]；T=192；动作维=20；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/kuavo/wholebody/w8a16.toml](pi05/kuavo/wholebody/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；原帧=[848, 480]；T=192；动作维=20；TP=8 | 8（运行器默认） | 未登记；模型专用模板 |

## qwen3

[配置说明](qwen3/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [qwen3/text/0_6b/fp16.toml](qwen3/text/0_6b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`QWEN3_DENSE_FP16_SUPPORTED` |
| [qwen3/text/0_6b/fp16_instruct.toml](qwen3/text/0_6b/fp16_instruct.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`QWEN3_DENSE_FP16_SUPPORTED` |
| [qwen3/text/0_6b/w4a16_text_head_rtn_g32.toml](qwen3/text/0_6b/w4a16_text_head_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/0_6b/w4a16_text_rtn_g32.toml](qwen3/text/0_6b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/0_6b/w8a16_text_head.toml](qwen3/text/0_6b/w8a16_text_head.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/14b/fp16.toml](qwen3/text/14b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/14b/w4a16_text_head_rtn_g32.toml](qwen3/text/14b/w4a16_text_head_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/14b/w4a16_text_rtn_g32.toml](qwen3/text/14b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/14b/w8a16_text_head.toml](qwen3/text/14b/w8a16_text_head.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/1_7b/fp16.toml](qwen3/text/1_7b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`QWEN3_DENSE_FP16_SUPPORTED` |
| [qwen3/text/1_7b/w4a16_text_head_rtn_g32.toml](qwen3/text/1_7b/w4a16_text_head_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/1_7b/w4a16_text_rtn_g32.toml](qwen3/text/1_7b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/1_7b/w8a16_text_head.toml](qwen3/text/1_7b/w8a16_text_head.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/32b/w4a16_text_head_rtn_g32.toml](qwen3/text/32b/w4a16_text_head_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/32b/w4a16_text_rtn_g32.toml](qwen3/text/32b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/32b/w8a16_text_head.toml](qwen3/text/32b/w8a16_text_head.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=128；D=7；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/4b/fp16.toml](qwen3/text/4b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`QWEN3_DENSE_FP16_SUPPORTED` |
| [qwen3/text/4b/fp16_instruct.toml](qwen3/text/4b/fp16_instruct.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`QWEN3_DENSE_FP16_SUPPORTED` |
| [qwen3/text/4b/w4a16_text_head_rtn_g32.toml](qwen3/text/4b/w4a16_text_head_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/4b/w4a16_text_rtn_g32.toml](qwen3/text/4b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/4b/w8a16_text_head.toml](qwen3/text/4b/w8a16_text_head.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/8b/fp16.toml](qwen3/text/8b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`QWEN3_DENSE_FP16_SUPPORTED` |
| [qwen3/text/8b/w4a16_text_head_rtn_g32.toml](qwen3/text/8b/w4a16_text_head_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/8b/w4a16_text_rtn_g32.toml](qwen3/text/8b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3/text/8b/w8a16_text_head.toml](qwen3/text/8b/w8a16_text_head.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |

## qwen3_5

[配置说明](qwen3_5/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [qwen3_5/text/0_8b/fp16.toml](qwen3_5/text/0_8b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`QWEN35_DENSE_FP16_TEXT_SUPPORTED` |
| [qwen3_5/text/0_8b/w4a16_text_rtn_g32.toml](qwen3_5/text/0_8b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/0_8b/w8a16_text.toml](qwen3_5/text/0_8b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/27b/fp16.toml](qwen3_5/text/27b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/27b/w4a16_mlp_activation_awq_g128.toml](qwen3_5/text/27b/w4a16_mlp_activation_awq_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 4 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/27b/w4a16_mlp_rtn_g32.toml](qwen3_5/text/27b/w4a16_mlp_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 4 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/27b/w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml](qwen3_5/text/27b/w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 4 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/27b/w4a16_text_rtn_g32.toml](qwen3_5/text/27b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 8 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/27b/w8a16_text.toml](qwen3_5/text/27b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4；TP=8 | 8 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/2b/fp16.toml](qwen3_5/text/2b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`QWEN35_DENSE_FP16_TEXT_SUPPORTED` |
| [qwen3_5/text/2b/w4a16_mlp_awq_g128.toml](qwen3_5/text/2b/w4a16_mlp_awq_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 4 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/2b/w4a16_mlp_rtn_g32.toml](qwen3_5/text/2b/w4a16_mlp_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 4 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml](qwen3_5/text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 4 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/2b/w4a16_text_rtn_g32.toml](qwen3_5/text/2b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/2b/w8a16_text.toml](qwen3_5/text/2b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn.toml](qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn.toml) | [run.py](../run.py)<br>运行模板 | `mixed-e4m3`；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128.toml](qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=31；TP=8 | 24 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128.toml](qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=31；TP=8 | 24 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/4b/fp16.toml](qwen3_5/text/4b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`QWEN35_DENSE_FP16_TEXT_SUPPORTED` |
| [qwen3_5/text/4b/w4a16_mlp_awq_g128.toml](qwen3_5/text/4b/w4a16_mlp_awq_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 4 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/4b/w4a16_text_rtn_g32.toml](qwen3_5/text/4b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/4b/w8a16_text.toml](qwen3_5/text/4b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/9b/fp16.toml](qwen3_5/text/9b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`QWEN35_DENSE_FP16_TEXT_SUPPORTED` |
| [qwen3_5/text/9b/w4a16_mlp_awq_g128.toml](qwen3_5/text/9b/w4a16_mlp_awq_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 4 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/9b/w4a16_text_rtn_g32.toml](qwen3_5/text/9b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/text/9b/w8a16_text.toml](qwen3_5/text/9b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/0_8b/fp16.toml](qwen3_5/vl/0_8b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/0_8b/w4a16_text_rtn_g32.toml](qwen3_5/vl/0_8b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=32；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/0_8b/w8a16_text.toml](qwen3_5/vl/0_8b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=32；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/27b/fp16.toml](qwen3_5/vl/27b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/27b/w4a16_text_rtn_g32.toml](qwen3_5/vl/27b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=32；TP=8 | 8 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/27b/w8a16_text.toml](qwen3_5/vl/27b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=32；TP=8 | 8 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/2b/fp16.toml](qwen3_5/vl/2b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/2b/w4a16_text_rtn_g32.toml](qwen3_5/vl/2b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=32；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/2b/w8a16_text.toml](qwen3_5/vl/2b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=32；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn.toml](qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn.toml) | [run.py](../run.py)<br>运行模板 | `mixed-e4m3`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/35b_a3b/w4a16_experts_gptq_g128.toml](qwen3_5/vl/35b_a3b/w4a16_experts_gptq_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 24 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/35b_a3b/w4a16_experts_rtn_g128.toml](qwen3_5/vl/35b_a3b/w4a16_experts_rtn_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 24 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/4b/fp16.toml](qwen3_5/vl/4b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/4b/w4a16_text_rtn_g32.toml](qwen3_5/vl/4b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=32；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/4b/w8a16_text.toml](qwen3_5/vl/4b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=32；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/9b/fp16.toml](qwen3_5/vl/9b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_5/vl/9b/w8a16_text.toml](qwen3_5/vl/9b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=32；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |

## qwen3_asr

[配置说明](qwen3_asr/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [qwen3_asr/0_6b/fp16.toml](qwen3_asr/0_6b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；TP=8 | 4 | 未登记；模型专用模板 |
| [qwen3_asr/0_6b/nvfp4.toml](qwen3_asr/0_6b/nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `nvfp4`；TP=8 | 4 | 未登记；模型专用模板 |
| [qwen3_asr/0_6b/w4a16_g32.toml](qwen3_asr/0_6b/w4a16_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16_g32`；TP=8 | 4 | 未登记；模型专用模板 |
| [qwen3_asr/0_6b/w8a16.toml](qwen3_asr/0_6b/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；TP=8 | 4 | 未登记；模型专用模板 |
| [qwen3_asr/1_7b/fp16.toml](qwen3_asr/1_7b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；TP=8 | 4 | 未登记；模型专用模板 |
| [qwen3_asr/1_7b/nvfp4.toml](qwen3_asr/1_7b/nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `nvfp4`；TP=8 | 4 | 未登记；模型专用模板 |
| [qwen3_asr/1_7b/w4a16_g32.toml](qwen3_asr/1_7b/w4a16_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16_g32`；TP=8 | 4 | 未登记；模型专用模板 |
| [qwen3_asr/1_7b/w8a16.toml](qwen3_asr/1_7b/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；TP=8 | 4 | 未登记；模型专用模板 |

## qwen3_tts

[配置说明](qwen3_tts/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [qwen3_tts/0_6b/fp16.toml](qwen3_tts/0_6b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；TP=8 | 16 | 未登记；模型专用模板 |
| [qwen3_tts/0_6b/fp16_cpu_codec.toml](qwen3_tts/0_6b/fp16_cpu_codec.toml) | [run.py](../run.py)<br>运行模板 | `—`；TP=8 | 16 | 未登记；模型专用模板 |
| [qwen3_tts/0_6b/nvfp4.toml](qwen3_tts/0_6b/nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `nvfp4`；TP=8 | 16 | 未登记；模型专用模板 |
| [qwen3_tts/0_6b/w4a16_g32.toml](qwen3_tts/0_6b/w4a16_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16_g32`；TP=8 | 16 | 未登记；模型专用模板 |
| [qwen3_tts/0_6b/w8a16.toml](qwen3_tts/0_6b/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；TP=8 | 16 | 未登记；模型专用模板 |
| [qwen3_tts/1_7b/fp16.toml](qwen3_tts/1_7b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；TP=8 | 16 | 未登记；模型专用模板 |
| [qwen3_tts/1_7b/nvfp4.toml](qwen3_tts/1_7b/nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `nvfp4`；TP=8 | 16 | 未登记；模型专用模板 |
| [qwen3_tts/1_7b/w4a16_g32.toml](qwen3_tts/1_7b/w4a16_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16_g32`；TP=8 | 16 | 未登记；模型专用模板 |
| [qwen3_tts/1_7b/w8a16.toml](qwen3_tts/1_7b/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；TP=8 | 16 | 未登记；模型专用模板 |

## qwen3_vl

[配置说明](qwen3_vl/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [qwen3_vl/text/2b/fp16.toml](qwen3_vl/text/2b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/text/2b/w4a16_text_awq_g32.toml](qwen3_vl/text/2b/w4a16_text_awq_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=512；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/text/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/text/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=512；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/text/2b/w8a16_text.toml](qwen3_vl/text/2b/w8a16_text.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=512；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/text/2b/w8a16_text_head_vision.toml](qwen3_vl/text/2b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=512；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/text/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml](qwen3_vl/text/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=512；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/text/32b/w8a16_text_head_vision.toml](qwen3_vl/text/32b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=512；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/text/4b/fp16.toml](qwen3_vl/text/4b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/text/4b/w4a16_text_awq_g32.toml](qwen3_vl/text/4b/w4a16_text_awq_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=512；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/text/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/text/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=512；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/text/4b/w8a16_text_head_vision.toml](qwen3_vl/text/4b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=512；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/text/8b/fp16.toml](qwen3_vl/text/8b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/text/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/text/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=512；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/text/8b/w8a16_text_head_vision.toml](qwen3_vl/text/8b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=512；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vision/2b/fp16.toml](qwen3_vl/vision/2b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vision/4b/fp16.toml](qwen3_vl/vision/4b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vision/8b/fp16.toml](qwen3_vl/vision/8b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/2b/fp16.toml](qwen3_vl/vl/2b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/2b/w4a16_text_awq_g32.toml](qwen3_vl/vl/2b/w4a16_text_awq_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=430；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/vl/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=430；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/2b/w8a16_text.toml](qwen3_vl/vl/2b/w8a16_text.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=430；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/2b/w8a16_text_head_vision.toml](qwen3_vl/vl/2b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=430；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml](qwen3_vl/vl/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=430；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/32b/w8a16_text.toml](qwen3_vl/vl/32b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/32b/w8a16_text_head_vision.toml](qwen3_vl/vl/32b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=430；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/4b/fp16.toml](qwen3_vl/vl/4b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/4b/w4a16_text_awq_g32.toml](qwen3_vl/vl/4b/w4a16_text_awq_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=430；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/vl/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=430；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/4b/w8a16_text_head_vision.toml](qwen3_vl/vl/4b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=430；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/8b/fp16.toml](qwen3_vl/vl/8b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/vl/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=430；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [qwen3_vl/vl/8b/w8a16_text_head_vision.toml](qwen3_vl/vl/8b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=430；D=31；TP=8 | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |

## rhinovla

[配置说明](rhinovla/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [rhinovla/v1/fp16.toml](rhinovla/v1/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16` | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [rhinovla/v3/controlled/fp16.toml](rhinovla/v3/controlled/fp16.toml) | [run.py](../run.py)<br>受控验证 | `fp16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/controlled/w8a16_action.toml](rhinovla/v3/controlled/w8a16_action.toml) | [run.py](../run.py)<br>受控验证 | `w8a16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/controlled/w8a16_pipeline.toml](rhinovla/v3/controlled/w8a16_pipeline.toml) | [run.py](../run.py)<br>受控验证 | `w8a16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/controlled/w8a16_pipeline_high.toml](rhinovla/v3/controlled/w8a16_pipeline_high.toml) | [run.py](../run.py)<br>受控验证 | `w8a16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/controlled/w8a16_pipeline_high_fused.toml](rhinovla/v3/controlled/w8a16_pipeline_high_fused.toml) | [run.py](../run.py)<br>受控验证 | `w8a16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_k.toml](rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_k.toml) | [run.py](../run.py)<br>受控验证 | `w8a16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_qk.toml](rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_qk.toml) | [run.py](../run.py)<br>受控验证 | `w8a16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_qk_partial_rope.toml](rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_qk_partial_rope.toml) | [run.py](../run.py)<br>受控验证 | `w8a16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/controlled/w8a16_pipeline_input_cache.toml](rhinovla/v3/controlled/w8a16_pipeline_input_cache.toml) | [run.py](../run.py)<br>受控验证 | `w8a16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/fp16.toml](rhinovla/v3/fp16.toml) | [run.py](../run.py)<br>调用者 factory | `fp16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/rtc_fp16.toml](rhinovla/v3/rtc_fp16.toml) | [run.py](../run.py)<br>调用者 factory | `fp16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/rtc_w8a16_action.toml](rhinovla/v3/rtc_w8a16_action.toml) | [run.py](../run.py)<br>调用者 factory | `w8a16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/rtc_w8a16_action_adarms.toml](rhinovla/v3/rtc_w8a16_action_adarms.toml) | [run.py](../run.py)<br>调用者 factory | `w8a16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/rtc_w8a16_pipeline.toml](rhinovla/v3/rtc_w8a16_pipeline.toml) | [run.py](../run.py)<br>调用者 factory | `w8a16` | 12 | 调用者 factory 模板 |
| [rhinovla/v3/rtc_w8a16_pipeline_high_fused_vector_qk.toml](rhinovla/v3/rtc_w8a16_pipeline_high_fused_vector_qk.toml) | [run.py](../run.py)<br>调用者 factory | `w8a16` | 12 | 调用者 factory 模板 |
| [rhinovla/v3/w8a16_action.toml](rhinovla/v3/w8a16_action.toml) | [run.py](../run.py)<br>调用者 factory | `w8a16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/w8a16_action_adarms.toml](rhinovla/v3/w8a16_action_adarms.toml) | [run.py](../run.py)<br>调用者 factory | `w8a16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/w8a16_pipeline.toml](rhinovla/v3/w8a16_pipeline.toml) | [run.py](../run.py)<br>调用者 factory | `w8a16` | 12 | 调用者 factory 模板 |
| [rhinovla/v3/w8a16_pipeline_high_fused_vector_qk.toml](rhinovla/v3/w8a16_pipeline_high_fused_vector_qk.toml) | [run.py](../run.py)<br>调用者 factory | `w8a16` | 12 | 调用者 factory 模板 |

## wall_oss

[配置说明](wall_oss/README.md) · [权重准备](WEIGHTS.md)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [wall_oss/base/2cam/224/fp16.toml](wall_oss/base/2cam/224/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；图像=[224, 224]；T=32 | 4 | 未登记；模型专用模板 |
| [wall_oss/base/2cam/224/w8a16.toml](wall_oss/base/2cam/224/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；图像=[224, 224]；T=32 | 8 | 未登记；模型专用模板 |
| [wall_oss/base/2cam/224/w8a16_action_nvfp4.toml](wall_oss/base/2cam/224/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_action_nvfp4`；相机=2；图像=[224, 224]；T=32 | 4 | 未登记；模型专用模板 |
| [wall_oss/base/2cam/256/fp16.toml](wall_oss/base/2cam/256/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；图像=[256, 256]；T=32 | 4 | 未登记；模型专用模板 |
| [wall_oss/base/2cam/256/w8a16.toml](wall_oss/base/2cam/256/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；图像=[256, 256]；T=32 | 8 | 未登记；模型专用模板 |
| [wall_oss/base/2cam/256/w8a16_action_nvfp4.toml](wall_oss/base/2cam/256/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_action_nvfp4`；相机=2；图像=[256, 256]；T=32 | 8 | 未登记；模型专用模板 |
| [wall_oss/base/3cam/224/fp16.toml](wall_oss/base/3cam/224/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；图像=[224, 224]；T=32 | 4 | 未登记；模型专用模板 |
| [wall_oss/base/3cam/224/w8a16.toml](wall_oss/base/3cam/224/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；图像=[224, 224]；T=32 | 12 | 未登记；模型专用模板 |
| [wall_oss/base/3cam/224/w8a16_action_nvfp4.toml](wall_oss/base/3cam/224/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_action_nvfp4`；相机=3；图像=[224, 224]；T=32 | 8 | 未登记；模型专用模板 |
| [wall_oss/base/3cam/256/fp16.toml](wall_oss/base/3cam/256/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；图像=[256, 256]；T=32 | 8 | 未登记；模型专用模板 |
| [wall_oss/base/3cam/256/w8a16.toml](wall_oss/base/3cam/256/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；图像=[256, 256]；T=32 | 12 | 未登记；模型专用模板 |
| [wall_oss/base/3cam/256/w8a16_action_nvfp4.toml](wall_oss/base/3cam/256/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_action_nvfp4`；相机=3；图像=[256, 256]；T=32 | 12 | 未登记；模型专用模板 |
| [wall_oss/legacy/fp16.toml](wall_oss/legacy/fp16.toml) | [run.py](../run.py)<br>历史复现 | `fp16` | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [wall_oss/legacy/w4a16_channel.toml](wall_oss/legacy/w4a16_channel.toml) | [run.py](../run.py)<br>历史复现 | `w4a16` | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [wall_oss/legacy/w4a16_g32.toml](wall_oss/legacy/w4a16_g32.toml) | [run.py](../run.py)<br>历史复现 | `w4a16-pgrp` | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |
| [wall_oss/legacy/w8a16.toml](wall_oss/legacy/w8a16.toml) | [run.py](../run.py)<br>历史复现 | `w8a16` | 继承 | `SUPPORTED`<br>`BOARD_MEASURED_CURRENT_CONFIG_SUPPORTED` |

合计 **215** 份配置。
