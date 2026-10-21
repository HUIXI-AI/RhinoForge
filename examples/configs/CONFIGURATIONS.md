# 完整配置索引

由 `python scripts/check_examples.py --write` 生成，覆盖所有 TOML。登记状态来自各配置的 profile。配置差异、权重来源和路径绑定见模型指南。

[选型与使用](README.md) · [登记支持矩阵](../../docs/model_matrix.zh.md) · [权重准备](WEIGHTS.md) · [通用参数](PARAMETERS.md)

CPU 列为请求值，`继承` 表示按运行器或环境设置。固定 P/D 模板指定输入长度、生成长度及计时方式。

## dinov3

[配置说明](dinov3/README.md) · [权重准备](dinov3/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [dinov3/vit_b/fp16.toml](dinov3/vit_b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16` | 继承 | `SUPPORTED`<br>`DINOV3_VIT_B_SUPPORTED` |

## g05

[配置说明](g05/README.md) · [权重准备](g05/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [g05/2cam/fp16.toml](g05/2cam/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；T列表=[32, 52, 64, 128, 192, 256] | 8 | 未登记；模型专用模板 |
| [g05/2cam/w8a16.toml](g05/2cam/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；T列表=[32, 52, 64, 128, 192, 256] | 8 | 未登记；模型专用模板 |
| [g05/2cam/w8a16_action_nvfp4.toml](g05/2cam/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_nvfp4`；相机=2；T列表=[32, 52, 64, 128, 192, 256] | 8 | 未登记；模型专用模板 |
| [g05/3cam/fp16.toml](g05/3cam/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；T列表=[32, 52, 64, 128, 192, 256] | 8 | 未登记；模型专用模板 |
| [g05/3cam/w8a16.toml](g05/3cam/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；T列表=[32, 52, 64, 128, 192, 256] | 8 | 未登记；模型专用模板 |
| [g05/3cam/w8a16_action_nvfp4.toml](g05/3cam/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_nvfp4`；相机=3；T列表=[32, 52, 64, 128, 192, 256] | 8 | 未登记；模型专用模板 |
| [g05/legacy/fp16.toml](g05/legacy/fp16.toml) | [run.py](../run.py)<br>历史复现 | `fp16` | 8（运行器默认） | `CONTROLLED_EVALUATION`<br>`G05_EXACT_K1_CONTROLLED_TEMPORARY_SUPPORT` |

## gemma4

[配置说明](gemma4/README.md) · [权重准备](gemma4/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [gemma4/e4b/fp16.toml](gemma4/e4b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`GEMMA4_E4B_TEXT_MVP_SUPPORTED` |

## gr00t

[配置说明](gr00t/README.md) · [权重准备](gr00t/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [gr00t/n1_7_droid/224/fp16.toml](gr00t/n1_7_droid/224/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；图像=224；T=32 | 8 | `SUPPORTED_LIMITED`<br>`GR00T_N1D7_DELIVERY_SUPPORTED` |
| [gr00t/n1_7_droid/224/w8a16.toml](gr00t/n1_7_droid/224/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；图像=224；T=32 | 8 | `SUPPORTED_LIMITED`<br>`GR00T_EXISTING_W8A16_DELIVERY_PATH` |
| [gr00t/n1_7_droid/256/fp16.toml](gr00t/n1_7_droid/256/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；图像=256；T=32 | 8 | `SUPPORTED_LIMITED`<br>`GR00T_N1D7_DELIVERY_SUPPORTED` |
| [gr00t/n1_7_droid/256/w8a16.toml](gr00t/n1_7_droid/256/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；图像=256；T=32 | 8 | `SUPPORTED_LIMITED`<br>`GR00T_EXISTING_W8A16_DELIVERY_PATH` |

## halo

[配置说明](halo/README.md) · [权重准备](halo/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [halo/action_expert/fp16.toml](halo/action_expert/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16` | 继承 | `SUPPORTED_LIMITED`<br>`HALO_SINGLE_ACTION_EXPERT_STEP_SUPPORTED` |

## hy_embodied

[配置说明](hy_embodied/README.md) · [权重准备](hy_embodied/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [hy_embodied/umi/fp16.toml](hy_embodied/umi/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16` | 继承 | `SUPPORTED_LIMITED`<br>`HY_VLA_IMMUTABLE_V2_FP16_SUPPORTED` |

## hy_vla

[配置说明](hy_vla/README.md) · [权重准备](hy_vla/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [hy_vla/2cam/fp16.toml](hy_vla/2cam/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；T=32；TP=8 | 12（运行器默认） | 未登记；模型专用模板 |
| [hy_vla/2cam/w8a16.toml](hy_vla/2cam/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；T=32；TP=8 | 12（运行器默认） | 未登记；模型专用模板 |
| [hy_vla/2cam/w8a16_action_nvfp4.toml](hy_vla/2cam/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_nvfp4`；相机=2；T=32；TP=8 | 12（运行器默认） | 未登记；模型专用模板 |
| [hy_vla/3cam/fp16.toml](hy_vla/3cam/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；T=32；TP=8 | 12（运行器默认） | 未登记；模型专用模板 |
| [hy_vla/3cam/w8a16.toml](hy_vla/3cam/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；T=32；TP=8 | 12（运行器默认） | 未登记；模型专用模板 |
| [hy_vla/3cam/w8a16_action_nvfp4.toml](hy_vla/3cam/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_nvfp4`；相机=3；T=32；TP=8 | 12（运行器默认） | 未登记；模型专用模板 |

## internvla

[配置说明](internvla/README.md) · [权重准备](internvla/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [internvla/navdp/w8a16_system2.toml](internvla/navdp/w8a16_system2.toml) | [run.py](../run.py)<br>运行模板 | `fp16` | 继承 | `CONTROLLED_EVALUATION`<br>`INTERNVLA_N1_PRIVATE_CONTROLLED_PAIR` |

## lingbot

[配置说明](lingbot/README.md) · [权重准备](lingbot/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [lingbot/legacy/exact_z2/6b/w8a16.toml](lingbot/legacy/exact_z2/6b/w8a16.toml) | [run.py](../run.py)<br>历史复现 | `w8a16` | 继承 | `CONTROLLED_EVALUATION`<br>`LINGBOT_V2_EXACT_Z2_CONTROLLED_TEMPORARY_SUPPORT` |
| [lingbot/v1/4b/fp16_cpu_vision.toml](lingbot/v1/4b/fp16_cpu_vision.toml) | [run.py](../run.py)<br>运行模板 | `cpu-fp32+rpu-fp16` | 继承 | `SUPPORTED_LIMITED`<br>`LINGBOT_V1_SHIPPED_MIXED_VISION_PATH` |
| [lingbot/v2/2cam_t128/fp16.toml](lingbot/v2/2cam_t128/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；有效T=128；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/2cam_t128/w8a16.toml](lingbot/v2/2cam_t128/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；有效T=128；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/2cam_t16/fp16.toml](lingbot/v2/2cam_t16/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；有效T=16；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/2cam_t16/w8a16.toml](lingbot/v2/2cam_t16/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；有效T=16；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/2cam_t160/fp16.toml](lingbot/v2/2cam_t160/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；有效T=160；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/2cam_t160/w8a16.toml](lingbot/v2/2cam_t160/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；有效T=160；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/2cam_t32/fp16.toml](lingbot/v2/2cam_t32/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；有效T=32；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/2cam_t32/w8a16.toml](lingbot/v2/2cam_t32/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；有效T=32；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/2cam_t64/fp16.toml](lingbot/v2/2cam_t64/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；有效T=64；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/2cam_t64/w8a16.toml](lingbot/v2/2cam_t64/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；有效T=64；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/3cam_t128/fp16.toml](lingbot/v2/3cam_t128/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；有效T=128；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/3cam_t128/w8a16.toml](lingbot/v2/3cam_t128/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；有效T=128；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/3cam_t16/fp16.toml](lingbot/v2/3cam_t16/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；有效T=16；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/3cam_t16/w8a16.toml](lingbot/v2/3cam_t16/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；有效T=16；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/3cam_t160/fp16.toml](lingbot/v2/3cam_t160/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；有效T=160；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/3cam_t160/w8a16.toml](lingbot/v2/3cam_t160/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；有效T=160；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/3cam_t3/fp16.toml](lingbot/v2/3cam_t3/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；有效T=3；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/3cam_t3/w8a16.toml](lingbot/v2/3cam_t3/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；有效T=3；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/3cam_t32/fp16.toml](lingbot/v2/3cam_t32/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；有效T=32；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/3cam_t32/w8a16.toml](lingbot/v2/3cam_t32/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；有效T=32；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/3cam_t64/fp16.toml](lingbot/v2/3cam_t64/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；有效T=64；TP=8 | 1 | 未登记；模型专用模板 |
| [lingbot/v2/3cam_t64/w8a16.toml](lingbot/v2/3cam_t64/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；有效T=64；TP=8 | 1 | 未登记；模型专用模板 |

## llama

[配置说明](llama/README.md) · [权重准备](llama/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [llama/3_2_1b/fp16.toml](llama/3_2_1b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED`<br>`LLAMA_3_2_1B_SUPPORTED` |

## pi05

[配置说明](pi05/README.md) · [权重准备](pi05/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [pi05/2cam/fp16.toml](pi05/2cam/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=2；T=32 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/2cam/w8a16.toml](pi05/2cam/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=2；T=32 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/2cam/w8a16_action_nvfp4.toml](pi05/2cam/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_action_nvfp4`；相机=2；T=32 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/2cam/w8a16_prefill_w8a8_action_nvfp4.toml](pi05/2cam/w8a16_prefill_w8a8_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_prefill_w8a8_action_nvfp4`；相机=2；T=32 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/3cam/fp16.toml](pi05/3cam/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；相机=3；T=32 | 8（运行器默认） | `SUPPORTED_LIMITED`<br>`PI05_LIBERO_FP16_SUPPORTED` |
| [pi05/3cam/w8a16.toml](pi05/3cam/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；相机=3；T=32 | 8（运行器默认） | `SUPPORTED_LIMITED`<br>`PI05_LIBERO_W8A16_SUPPORTED_LIMITED` |
| [pi05/3cam/w8a16_action_nvfp4.toml](pi05/3cam/w8a16_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_action_nvfp4`；相机=3；T=32 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/3cam/w8a16_prefill_w8a8_action_nvfp4.toml](pi05/3cam/w8a16_prefill_w8a8_action_nvfp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_prefill_w8a8_action_nvfp4`；相机=3；T=32 | 8（运行器默认） | 未登记；模型专用模板 |
| [pi05/generic/3cam/fp16_tp4.toml](pi05/generic/3cam/fp16_tp4.toml) | [run.py](../run.py)<br>通用执行 | `fp16`；相机=3；T=32；TP=4 | 8（运行器默认） | `IMPLEMENTED_UNVERIFIED_HW`<br>`PI05_LIBERO_FP16_TP4_UNVERIFIED_HW` |
| [pi05/generic/3cam/fp16_tp6.toml](pi05/generic/3cam/fp16_tp6.toml) | [run.py](../run.py)<br>通用执行 | `fp16`；相机=3；T=32；TP=6 | 8（运行器默认） | `IMPLEMENTED_UNVERIFIED_HW`<br>`PI05_LIBERO_FP16_TP6_UNVERIFIED_HW` |
| [pi05/generic/3cam/w8a16_tp4.toml](pi05/generic/3cam/w8a16_tp4.toml) | [run.py](../run.py)<br>通用执行 | `w8a16`；相机=3；T=32；TP=4 | 8（运行器默认） | `IMPLEMENTED_UNVERIFIED_HW`<br>`PI05_LIBERO_W8A16_TP4_UNVERIFIED_HW` |
| [pi05/generic/3cam/w8a16_tp6.toml](pi05/generic/3cam/w8a16_tp6.toml) | [run.py](../run.py)<br>通用执行 | `w8a16`；相机=3；T=32；TP=6 | 8（运行器默认） | `IMPLEMENTED_UNVERIFIED_HW`<br>`PI05_LIBERO_W8A16_TP6_UNVERIFIED_HW` |

## qwen3

[配置说明](qwen3/README.md) · [权重准备](qwen3/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [qwen3/text/0_6b/fp16.toml](qwen3/text/0_6b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED`<br>`QWEN3_DENSE_FP16_SUPPORTED` |
| [qwen3/text/0_6b/fp16_instruct.toml](qwen3/text/0_6b/fp16_instruct.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED`<br>`QWEN3_DENSE_FP16_SUPPORTED` |
| [qwen3/text/0_6b/w4a16_text_head_rtn_g32.toml](qwen3/text/0_6b/w4a16_text_head_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_DENSE_W4_GROUP32_CONTROLLED_UNCERTIFIED` |
| [qwen3/text/0_6b/w4a16_text_rtn_g32.toml](qwen3/text/0_6b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_0_6B_W4_GROUP32_CONTROLLED_UNCERTIFIED` |
| [qwen3/text/0_6b/w8a16_text_head.toml](qwen3/text/0_6b/w8a16_text_head.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_W8A16_PROFILE_SUPPORTED` |
| [qwen3/text/14b/w4a16_text_head_rtn_g32.toml](qwen3/text/14b/w4a16_text_head_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_DENSE_W4_GROUP32_CONTROLLED_UNCERTIFIED` |
| [qwen3/text/14b/w4a16_text_rtn_g32.toml](qwen3/text/14b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_DENSE_W4_GROUP32_CONTROLLED_UNCERTIFIED` |
| [qwen3/text/14b/w8a16_text_head.toml](qwen3/text/14b/w8a16_text_head.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_W8A16_PROFILE_SUPPORTED` |
| [qwen3/text/1_7b/fp16.toml](qwen3/text/1_7b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED`<br>`QWEN3_DENSE_FP16_SUPPORTED` |
| [qwen3/text/1_7b/w4a16_text_head_rtn_g32.toml](qwen3/text/1_7b/w4a16_text_head_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_DENSE_W4_GROUP32_CONTROLLED_UNCERTIFIED` |
| [qwen3/text/1_7b/w4a16_text_rtn_g32.toml](qwen3/text/1_7b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_DENSE_W4_GROUP32_CONTROLLED_UNCERTIFIED` |
| [qwen3/text/1_7b/w8a16_text_head.toml](qwen3/text/1_7b/w8a16_text_head.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_W8A16_PROFILE_SUPPORTED` |
| [qwen3/text/32b/w4a16_text_head_rtn_g32.toml](qwen3/text/32b/w4a16_text_head_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_DENSE_W4_GROUP32_CONTROLLED_UNCERTIFIED` |
| [qwen3/text/32b/w4a16_text_rtn_g32.toml](qwen3/text/32b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_DENSE_W4_GROUP32_CONTROLLED_UNCERTIFIED` |
| [qwen3/text/32b/w8a16_text_head.toml](qwen3/text/32b/w8a16_text_head.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=128；D=7；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_32B_W8_CONTROLLED_UNCERTIFIED` |
| [qwen3/text/4b/fp16.toml](qwen3/text/4b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED`<br>`QWEN3_DENSE_FP16_SUPPORTED` |
| [qwen3/text/4b/fp16_instruct.toml](qwen3/text/4b/fp16_instruct.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED`<br>`QWEN3_DENSE_FP16_SUPPORTED` |
| [qwen3/text/4b/w4a16_text_head_rtn_g32.toml](qwen3/text/4b/w4a16_text_head_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_DENSE_W4_GROUP32_CONTROLLED_UNCERTIFIED` |
| [qwen3/text/4b/w4a16_text_rtn_g32.toml](qwen3/text/4b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_DENSE_W4_GROUP32_CONTROLLED_UNCERTIFIED` |
| [qwen3/text/4b/w8a16_text_head.toml](qwen3/text/4b/w8a16_text_head.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_W8A16_PROFILE_SUPPORTED` |
| [qwen3/text/8b/fp16.toml](qwen3/text/8b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED`<br>`QWEN3_DENSE_FP16_SUPPORTED` |
| [qwen3/text/8b/w4a16_text_head_rtn_g32.toml](qwen3/text/8b/w4a16_text_head_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_DENSE_W4_GROUP32_CONTROLLED_UNCERTIFIED` |
| [qwen3/text/8b/w4a16_text_rtn_g32.toml](qwen3/text/8b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=128；D=7；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_DENSE_W4_GROUP32_CONTROLLED_UNCERTIFIED` |
| [qwen3/text/8b/w8a16_text_head.toml](qwen3/text/8b/w8a16_text_head.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_W8A16_PROFILE_SUPPORTED` |

## qwen3_5

[配置说明](qwen3_5/README.md) · [权重准备](qwen3_5/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [qwen3_5/legacy/27b/w8a16_attn_mlp.toml](qwen3_5/legacy/27b/w8a16_attn_mlp.toml) | [run.py](../run.py)<br>历史复现 | `w8a16`；D=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN38_27B_W8_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/legacy/27b/w8a16_attn_mlp_tp4.toml](qwen3_5/legacy/27b/w8a16_attn_mlp_tp4.toml) | [run.py](../run.py)<br>历史复现 | `w8a16`；D=31；TP=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN38_27B_W8_TP4_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/legacy/27b/w8a16_attn_mlp_tp6.toml](qwen3_5/legacy/27b/w8a16_attn_mlp_tp6.toml) | [run.py](../run.py)<br>历史复现 | `w8a16`；D=31；TP=6 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN38_27B_W8_TP6_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/0_8b/fp16.toml](qwen3_5/text/0_8b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED`<br>`QWEN35_DENSE_FP16_TEXT_SUPPORTED` |
| [qwen3_5/text/0_8b/fp16_p4096_d4.toml](qwen3_5/text/0_8b/fp16_p4096_d4.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=4096；D=4 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_LONG_PREFILL_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/0_8b/fp16_p4096_d4096.toml](qwen3_5/text/0_8b/fp16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=4096；D=4096；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_LONG_PREFILL_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/0_8b/fp16_p64_d4096.toml](qwen3_5/text/0_8b/fp16_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=64；D=4096；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_LONG_PREFILL_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/0_8b/fp16_tp4.toml](qwen3_5/text/0_8b/fp16_tp4.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=32；TP=4 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_DENSE_TEXT_TP4_UNVERIFIED_HW` |
| [qwen3_5/text/0_8b/fp16_tp6.toml](qwen3_5/text/0_8b/fp16_tp6.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=32；TP=6 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_DENSE_TEXT_TP6_UNVERIFIED_HW` |
| [qwen3_5/text/0_8b/w4a16_text_rtn_g32.toml](qwen3_5/text/0_8b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/0_8b/w4a16_text_rtn_g32_p4096_d4096.toml](qwen3_5/text/0_8b/w4a16_text_rtn_g32_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=4096；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/0_8b/w4a16_text_rtn_g32_p64_d4096.toml](qwen3_5/text/0_8b/w4a16_text_rtn_g32_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=64；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/0_8b/w8a16_text.toml](qwen3_5/text/0_8b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/0_8b/w8a16_text_p4096_d4096.toml](qwen3_5/text/0_8b/w8a16_text_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=4096；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/0_8b/w8a16_text_p64_d4096.toml](qwen3_5/text/0_8b/w8a16_text_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=64；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/27b/w4a16_text_rtn_g32.toml](qwen3_5/text/27b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/text/27b/w4a16_text_rtn_g32_acc32_text.toml](qwen3_5/text/27b/w4a16_text_rtn_g32_acc32_text.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/text/27b/w4a16_text_rtn_g32_p4096_d4096.toml](qwen3_5/text/27b/w4a16_text_rtn_g32_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=4096；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/text/27b/w4a16_text_rtn_g32_p64_d4096.toml](qwen3_5/text/27b/w4a16_text_rtn_g32_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=64；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/text/27b/w8a16_text.toml](qwen3_5/text/27b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4；TP=8 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/text/27b/w8a16_text_acc32_text_p29_d31.toml](qwen3_5/text/27b/w8a16_text_acc32_text_p29_d31.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=29；D=31；TP=8 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/text/27b/w8a16_text_p29_d31.toml](qwen3_5/text/27b/w8a16_text_p29_d31.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=29；D=31；TP=8 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/text/27b/w8a16_text_p4096_d4.toml](qwen3_5/text/27b/w8a16_text_p4096_d4.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=4096；D=4；TP=8 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/text/27b/w8a16_text_p4096_d4096.toml](qwen3_5/text/27b/w8a16_text_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=4096；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/text/27b/w8a16_text_p64_d4096.toml](qwen3_5/text/27b/w8a16_text_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=64；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/text/27b/w8a16_text_tp4.toml](qwen3_5/text/27b/w8a16_text_tp4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=32；TP=4 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/text/27b/w8a16_text_tp4_p29_d31.toml](qwen3_5/text/27b/w8a16_text_tp4_p29_d31.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=29；D=31；TP=4 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/text/27b/w8a16_text_tp6.toml](qwen3_5/text/27b/w8a16_text_tp6.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=32；TP=6 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/text/27b/w8a16_text_tp6_p29_d31.toml](qwen3_5/text/27b/w8a16_text_tp6_p29_d31.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=29；D=31；TP=6 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/text/2b/fp16.toml](qwen3_5/text/2b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED`<br>`QWEN35_DENSE_FP16_TEXT_SUPPORTED` |
| [qwen3_5/text/2b/fp16_chunk640_p4096_d4.toml](qwen3_5/text/2b/fp16_chunk640_p4096_d4.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=4096；D=4 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_LONG_PREFILL_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/fp16_p4096_d4096.toml](qwen3_5/text/2b/fp16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=4096；D=4096；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_LONG_PREFILL_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/fp16_p64_d4096.toml](qwen3_5/text/2b/fp16_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=64；D=4096；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_LONG_PREFILL_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/fp16_tp4.toml](qwen3_5/text/2b/fp16_tp4.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=32；TP=4 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_DENSE_TEXT_TP4_UNVERIFIED_HW` |
| [qwen3_5/text/2b/fp16_tp6.toml](qwen3_5/text/2b/fp16_tp6.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=32；TP=6 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_DENSE_TEXT_TP6_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w4a16_mlp_awq_g128.toml](qwen3_5/text/2b/w4a16_mlp_awq_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w4a16_mlp_awq_g128_p4096_d4096.toml](qwen3_5/text/2b/w4a16_mlp_awq_g128_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=4096；D=4096；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w4a16_mlp_awq_g128_p64_d4096.toml](qwen3_5/text/2b/w4a16_mlp_awq_g128_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=64；D=4096；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w4a16_mlp_rtn_g32.toml](qwen3_5/text/2b/w4a16_mlp_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml](qwen3_5/text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16_p4096_d4096.toml](qwen3_5/text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=4096；D=4096；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16_p64_d4096.toml](qwen3_5/text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=64；D=4096；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w4a16_mlp_rtn_g32_p4096_d4096.toml](qwen3_5/text/2b/w4a16_mlp_rtn_g32_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=4096；D=4096；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w4a16_mlp_rtn_g32_p64_d4096.toml](qwen3_5/text/2b/w4a16_mlp_rtn_g32_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=64；D=4096；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w4a16_text_rtn_g32.toml](qwen3_5/text/2b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w4a16_text_rtn_g32_p4096_d4096.toml](qwen3_5/text/2b/w4a16_text_rtn_g32_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=4096；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w4a16_text_rtn_g32_p64_d4096.toml](qwen3_5/text/2b/w4a16_text_rtn_g32_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=64；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w8a16_text.toml](qwen3_5/text/2b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w8a16_text_p4096_d4096.toml](qwen3_5/text/2b/w8a16_text_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=4096；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/2b/w8a16_text_p64_d4096.toml](qwen3_5/text/2b/w8a16_text_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=64；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn.toml](qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn.toml) | [run.py](../run.py)<br>运行模板 | `mixed-e4m3`；D=31；TP=8 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_TEXT_TP8_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_p4096_d4096.toml](qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `mixed-e4m3`；P=4096；D=4096；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_TEXT_TP8_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_p64_d4096.toml](qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `mixed-e4m3`；P=64；D=4096；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_TEXT_TP8_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_tp4.toml](qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_tp4.toml) | [run.py](../run.py)<br>运行模板 | `mixed-e4m3`；D=31；TP=4 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_TEXT_TP4_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_tp6.toml](qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_tp6.toml) | [run.py](../run.py)<br>运行模板 | `mixed-e4m3`；D=31；TP=6 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_TEXT_TP6_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128.toml](qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=31；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_GPTQ_TEXT_TP8_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128_p4096_d4096.toml](qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=4096；D=4096；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_GPTQ_TEXT_TP8_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128_p64_d4096.toml](qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=64；D=4096；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_GPTQ_TEXT_TP8_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128.toml](qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=31；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_RTN_TEXT_TP8_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128_p4096_d4096.toml](qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=4096；D=4096；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_RTN_TEXT_TP8_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128_p64_d4096.toml](qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=64；D=4096；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_RTN_TEXT_TP8_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/4b/fp16.toml](qwen3_5/text/4b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED`<br>`QWEN35_DENSE_FP16_TEXT_SUPPORTED` |
| [qwen3_5/text/4b/fp16_p4096_d4.toml](qwen3_5/text/4b/fp16_p4096_d4.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=4096；D=4 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_LONG_PREFILL_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/4b/fp16_p4096_d4096.toml](qwen3_5/text/4b/fp16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=4096；D=4096；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_LONG_PREFILL_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/4b/fp16_p64_d4096.toml](qwen3_5/text/4b/fp16_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=64；D=4096；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_LONG_PREFILL_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/4b/fp16_tp4.toml](qwen3_5/text/4b/fp16_tp4.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=32；TP=4 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_DENSE_TEXT_TP4_UNVERIFIED_HW` |
| [qwen3_5/text/4b/fp16_tp6.toml](qwen3_5/text/4b/fp16_tp6.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=32；TP=6 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_DENSE_TEXT_TP6_UNVERIFIED_HW` |
| [qwen3_5/text/4b/w4a16_mlp_awq_g128.toml](qwen3_5/text/4b/w4a16_mlp_awq_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/4b/w4a16_mlp_awq_g128_p4096_d4096.toml](qwen3_5/text/4b/w4a16_mlp_awq_g128_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=4096；D=4096；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/4b/w4a16_mlp_awq_g128_p64_d4096.toml](qwen3_5/text/4b/w4a16_mlp_awq_g128_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=64；D=4096；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/4b/w4a16_text_rtn_g32.toml](qwen3_5/text/4b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/4b/w4a16_text_rtn_g32_p4096_d4096.toml](qwen3_5/text/4b/w4a16_text_rtn_g32_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=4096；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/4b/w4a16_text_rtn_g32_p64_d4096.toml](qwen3_5/text/4b/w4a16_text_rtn_g32_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=64；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/4b/w8a16_text.toml](qwen3_5/text/4b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/4b/w8a16_text_p4096_d4096.toml](qwen3_5/text/4b/w8a16_text_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=4096；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/4b/w8a16_text_p64_d4096.toml](qwen3_5/text/4b/w8a16_text_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=64；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/9b/fp16.toml](qwen3_5/text/9b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED`<br>`QWEN35_DENSE_FP16_TEXT_SUPPORTED` |
| [qwen3_5/text/9b/fp16_p4096_d4.toml](qwen3_5/text/9b/fp16_p4096_d4.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=4096；D=4 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_LONG_PREFILL_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/9b/fp16_p4096_d4096.toml](qwen3_5/text/9b/fp16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=4096；D=4096；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_LONG_PREFILL_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/9b/fp16_p64_d4096.toml](qwen3_5/text/9b/fp16_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=64；D=4096；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_LONG_PREFILL_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/9b/fp16_tp4.toml](qwen3_5/text/9b/fp16_tp4.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=32；TP=4 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_DENSE_TEXT_TP4_UNVERIFIED_HW` |
| [qwen3_5/text/9b/fp16_tp6.toml](qwen3_5/text/9b/fp16_tp6.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=32；TP=6 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_DENSE_TEXT_TP6_UNVERIFIED_HW` |
| [qwen3_5/text/9b/w4a16_mlp_awq_g128.toml](qwen3_5/text/9b/w4a16_mlp_awq_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/9b/w4a16_mlp_awq_g128_p4096_d4096.toml](qwen3_5/text/9b/w4a16_mlp_awq_g128_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=4096；D=4096；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/9b/w4a16_mlp_awq_g128_p64_d4096.toml](qwen3_5/text/9b/w4a16_mlp_awq_g128_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=64；D=4096；TP=8 | 4 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/9b/w4a16_text_rtn_g32.toml](qwen3_5/text/9b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/9b/w4a16_text_rtn_g32_p4096_d4096.toml](qwen3_5/text/9b/w4a16_text_rtn_g32_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=4096；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/9b/w4a16_text_rtn_g32_p64_d4096.toml](qwen3_5/text/9b/w4a16_text_rtn_g32_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=64；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/9b/w8a16_text.toml](qwen3_5/text/9b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/9b/w8a16_text_p4096_d4096.toml](qwen3_5/text/9b/w8a16_text_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=4096；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/text/9b/w8a16_text_p64_d4096.toml](qwen3_5/text/9b/w8a16_text_p64_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=64；D=4096；TP=8 | 24 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/vl/0_8b/fp16.toml](qwen3_5/vl/0_8b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_VISION_HW_UNVERIFIED` |
| [qwen3_5/vl/0_8b/fp16_p4096_d4096.toml](qwen3_5/vl/0_8b/fp16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；图像=448；P=4096；D=4096；TP=8 | 12 | `CONTROLLED_EVALUATION`<br>`QWEN35_VISION_HW_UNVERIFIED` |
| [qwen3_5/vl/0_8b/fp16_p512_d192.toml](qwen3_5/vl/0_8b/fp16_p512_d192.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；图像=448；P=512；D=192；TP=8 | 12 | `CONTROLLED_EVALUATION`<br>`QWEN35_VISION_HW_UNVERIFIED` |
| [qwen3_5/vl/0_8b/w4a16_text_rtn_g32.toml](qwen3_5/vl/0_8b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=32；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_VISION_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/vl/0_8b/w8a16_text.toml](qwen3_5/vl/0_8b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=32；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_VISION_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/vl/27b/w4a16_text_rtn_g32.toml](qwen3_5/vl/27b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=32；TP=8 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/vl/27b/w4a16_text_rtn_g32_acc32_text_vision.toml](qwen3_5/vl/27b/w4a16_text_rtn_g32_acc32_text_vision.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=32；TP=8 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/vl/27b/w8a16_text.toml](qwen3_5/vl/27b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=32；TP=8 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/vl/27b/w8a16_text_acc32_text.toml](qwen3_5/vl/27b/w8a16_text_acc32_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=32；TP=8 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/vl/27b/w8a16_text_acc32_vision.toml](qwen3_5/vl/27b/w8a16_text_acc32_vision.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=32；TP=8 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/vl/27b/w8a16_text_d4.toml](qwen3_5/vl/27b/w8a16_text_d4.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4；TP=8 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/vl/27b/w8a16_text_image_mascot_d31.toml](qwen3_5/vl/27b/w8a16_text_image_mascot_d31.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=31；TP=8 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/vl/27b/w8a16_text_p4096_d4096.toml](qwen3_5/vl/27b/w8a16_text_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；图像=448；P=4096；D=4096；TP=8 | 12 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/vl/27b/w8a16_text_p512_d192.toml](qwen3_5/vl/27b/w8a16_text_p512_d192.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；图像=448；P=512；D=192；TP=8 | 12 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/vl/27b/w8a16_text_tp4_image_mascot_d31.toml](qwen3_5/vl/27b/w8a16_text_tp4_image_mascot_d31.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=31；TP=4 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/vl/27b/w8a16_text_tp6_image_mascot_d31.toml](qwen3_5/vl/27b/w8a16_text_tp6_image_mascot_d31.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=31；TP=6 | 8 | `CONTROLLED_EVALUATION`<br>`QWEN35_OFFICIAL_27B_MEASURED_CONTROLLED` |
| [qwen3_5/vl/2b/fp16.toml](qwen3_5/vl/2b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_VISION_NUMERIC_BLOCKED` |
| [qwen3_5/vl/2b/fp16_p4096_d4096.toml](qwen3_5/vl/2b/fp16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；图像=448；P=4096；D=4096；TP=8 | 12 | `CONTROLLED_EVALUATION`<br>`QWEN35_VISION_NUMERIC_BLOCKED` |
| [qwen3_5/vl/2b/fp16_p512_d192.toml](qwen3_5/vl/2b/fp16_p512_d192.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；图像=448；P=512；D=192；TP=8 | 12 | `CONTROLLED_EVALUATION`<br>`QWEN35_VISION_NUMERIC_BLOCKED` |
| [qwen3_5/vl/2b/w4a16_text_rtn_g32.toml](qwen3_5/vl/2b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=32；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_VISION_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/vl/2b/w8a16_text.toml](qwen3_5/vl/2b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=32；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_VISION_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn.toml](qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn.toml) | [run.py](../run.py)<br>运行模板 | `mixed-e4m3`；D=4；TP=8 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_TP8_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn_p4096_d4096.toml](qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `mixed-e4m3`；P=4096；D=4096；TP=8 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_TP8_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn_tp4.toml](qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn_tp4.toml) | [run.py](../run.py)<br>运行模板 | `mixed-e4m3`；D=4；TP=4 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_TP4_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn_tp6.toml](qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn_tp6.toml) | [run.py](../run.py)<br>运行模板 | `mixed-e4m3`；D=4；TP=6 | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_TP6_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/vl/35b_a3b/w4a16_experts_gptq_g128.toml](qwen3_5/vl/35b_a3b/w4a16_experts_gptq_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_GPTQ_MULTIMODAL_TP8_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/vl/35b_a3b/w4a16_experts_rtn_g128.toml](qwen3_5/vl/35b_a3b/w4a16_experts_rtn_g128.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=4；TP=8 | 24 | `IMPLEMENTED_UNVERIFIED_HW`<br>`QWEN35_MOE_RTN_MULTIMODAL_TP8_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/vl/4b/fp16.toml](qwen3_5/vl/4b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_VISION_NUMERIC_BLOCKED` |
| [qwen3_5/vl/4b/fp16_p4096_d4096.toml](qwen3_5/vl/4b/fp16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；图像=448；P=4096；D=4096；TP=8 | 12 | `CONTROLLED_EVALUATION`<br>`QWEN35_VISION_NUMERIC_BLOCKED` |
| [qwen3_5/vl/4b/fp16_p512_d192.toml](qwen3_5/vl/4b/fp16_p512_d192.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；图像=448；P=512；D=192；TP=8 | 12 | `CONTROLLED_EVALUATION`<br>`QWEN35_VISION_NUMERIC_BLOCKED` |
| [qwen3_5/vl/4b/w4a16_text_rtn_g32.toml](qwen3_5/vl/4b/w4a16_text_rtn_g32.toml) | [run.py](../run.py)<br>运行模板 | `w4a16`；D=32；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_VISION_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/vl/4b/w8a16_text.toml](qwen3_5/vl/4b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=32；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_VISION_IMPLEMENTED_UNVERIFIED_HW` |
| [qwen3_5/vl/9b/fp16.toml](qwen3_5/vl/9b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_VISION_HW_UNVERIFIED` |
| [qwen3_5/vl/9b/fp16_p4096_d4096.toml](qwen3_5/vl/9b/fp16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；图像=448；P=4096；D=4096；TP=8 | 12 | `CONTROLLED_EVALUATION`<br>`QWEN35_VISION_HW_UNVERIFIED` |
| [qwen3_5/vl/9b/fp16_p512_d192.toml](qwen3_5/vl/9b/fp16_p512_d192.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；图像=448；P=512；D=192；TP=8 | 12 | `CONTROLLED_EVALUATION`<br>`QWEN35_VISION_HW_UNVERIFIED` |
| [qwen3_5/vl/9b/w8a16_text.toml](qwen3_5/vl/9b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=32；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN35_QUANT_VISION_IMPLEMENTED_UNVERIFIED_HW` |

## qwen3_asr

[配置说明](qwen3_asr/README.md) · [权重准备](qwen3_asr/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [qwen3_asr/0_6b/fp16.toml](qwen3_asr/0_6b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `—` | 4 | 未登记；模型专用模板 |

## qwen3_tts

[配置说明](qwen3_tts/README.md) · [权重准备](qwen3_tts/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [qwen3_tts/0_6b/base.toml](qwen3_tts/0_6b/base.toml) | [run.py](../run.py)<br>运行模板 | `—` | 16 | 未登记；模型专用模板 |
| [qwen3_tts/0_6b/base_rpu_codec.toml](qwen3_tts/0_6b/base_rpu_codec.toml) | [run.py](../run.py)<br>运行模板 | `—` | 16 | 未登记；模型专用模板 |

## qwen3_vl

[配置说明](qwen3_vl/README.md) · [权重准备](qwen3_vl/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [qwen3_vl/text/2b/fp16.toml](qwen3_vl/text/2b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_2B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/text/2b/fp16_p192_d192.toml](qwen3_vl/text/2b/fp16_p192_d192.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=192；D=192 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_2B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/text/2b/fp16_p4096_d4096.toml](qwen3_vl/text/2b/fp16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=4096；D=4096 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_2B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/text/2b/w4a16_text_awq_g32.toml](qwen3_vl/text/2b/w4a16_text_awq_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=512；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_2B_TEXT_W4_FP16_HEAD_VISION_CONTROLLED` |
| [qwen3_vl/text/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/text/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=512；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_2B_RUNTIME_W4_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/text/2b/w8a16_text.toml](qwen3_vl/text/2b/w8a16_text.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=512；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_2B_TEXT_W8_FP16_HEAD_VISION_CONTROLLED` |
| [qwen3_vl/text/2b/w8a16_text_head_vision.toml](qwen3_vl/text/2b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=512；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_2B_RUNTIME_W8_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/text/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml](qwen3_vl/text/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=512；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_32B_RUNTIME_W4_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/text/32b/w8a16_text_head_vision.toml](qwen3_vl/text/32b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=512；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_32B_RUNTIME_W8_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/text/4b/fp16.toml](qwen3_vl/text/4b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_4B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/text/4b/fp16_p192_d192.toml](qwen3_vl/text/4b/fp16_p192_d192.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=192；D=192 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_4B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/text/4b/fp16_p4096_d4096.toml](qwen3_vl/text/4b/fp16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=4096；D=4096；TP=8 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_4B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/text/4b/w4a16_text_awq_g32.toml](qwen3_vl/text/4b/w4a16_text_awq_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=512；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_4B_TEXT_W4_FP16_HEAD_VISION_CONTROLLED` |
| [qwen3_vl/text/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/text/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=512；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_4B_RUNTIME_W4_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/text/4b/w8a16_text_head_vision.toml](qwen3_vl/text/4b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=512；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_4B_RUNTIME_W8_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/text/8b/fp16.toml](qwen3_vl/text/8b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_8B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/text/8b/fp16_p192_d192.toml](qwen3_vl/text/8b/fp16_p192_d192.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=192；D=192 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_8B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/text/8b/fp16_p4096_d4096.toml](qwen3_vl/text/8b/fp16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=4096；D=4096；TP=8 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_8B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/text/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/text/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=512；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_8B_RUNTIME_W4_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/text/8b/w8a16_text_head_vision.toml](qwen3_vl/text/8b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=512；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_8B_RUNTIME_W8_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/vision/2b/fp16.toml](qwen3_vl/vision/2b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_2B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/vision/4b/fp16.toml](qwen3_vl/vision/4b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_4B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/vision/8b/fp16.toml](qwen3_vl/vision/8b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_8B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/vl/2b/fp16.toml](qwen3_vl/vl/2b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_2B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/vl/2b/fp16_p256_d192.toml](qwen3_vl/vl/2b/fp16_p256_d192.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=256；D=192 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_2B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/vl/2b/fp16_p4096_d4096.toml](qwen3_vl/vl/2b/fp16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=4096；D=4096 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_2B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/vl/2b/w4a16_text_awq_g32.toml](qwen3_vl/vl/2b/w4a16_text_awq_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=430；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_2B_TEXT_W4_FP16_HEAD_VISION_CONTROLLED` |
| [qwen3_vl/vl/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/vl/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=430；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_2B_RUNTIME_W4_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/vl/2b/w8a16_text.toml](qwen3_vl/vl/2b/w8a16_text.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=430；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_2B_TEXT_W8_FP16_HEAD_VISION_CONTROLLED` |
| [qwen3_vl/vl/2b/w8a16_text_head_vision.toml](qwen3_vl/vl/2b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=430；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_2B_RUNTIME_W8_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/vl/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml](qwen3_vl/vl/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=430；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_32B_RUNTIME_W4_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/vl/32b/w8a16_text.toml](qwen3_vl/vl/32b/w8a16_text.toml) | [run.py](../run.py)<br>运行模板 | `w8a16`；D=4 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_32B_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/vl/32b/w8a16_text_head_vision.toml](qwen3_vl/vl/32b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=430；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_32B_RUNTIME_W8_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/vl/4b/fp16.toml](qwen3_vl/vl/4b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_4B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/vl/4b/fp16_p256_d192.toml](qwen3_vl/vl/4b/fp16_p256_d192.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=256；D=192 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_4B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/vl/4b/fp16_p4096_d4096.toml](qwen3_vl/vl/4b/fp16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=4096；D=4096；TP=8 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_4B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/vl/4b/w4a16_text_awq_g32.toml](qwen3_vl/vl/4b/w4a16_text_awq_g32.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=430；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_4B_TEXT_W4_FP16_HEAD_VISION_CONTROLLED` |
| [qwen3_vl/vl/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/vl/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=430；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_4B_RUNTIME_W4_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/vl/4b/w8a16_text_head_vision.toml](qwen3_vl/vl/4b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=430；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_4B_RUNTIME_W8_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/vl/8b/fp16.toml](qwen3_vl/vl/8b/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16`；D=4 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_8B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/vl/8b/fp16_p256_d192.toml](qwen3_vl/vl/8b/fp16_p256_d192.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=256；D=192 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_8B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/vl/8b/fp16_p4096_d4096.toml](qwen3_vl/vl/8b/fp16_p4096_d4096.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `fp16`；P=4096；D=4096；TP=8 | 继承 | `SUPPORTED_LIMITED`<br>`QWEN3_VL_8B_MEASURED_PROFILE_SUPPORTED` |
| [qwen3_vl/vl/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](qwen3_vl/vl/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w4a16`；P=430；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_8B_RUNTIME_W4_CONTROLLED_EVALUATION_UNCERTIFIED` |
| [qwen3_vl/vl/8b/w8a16_text_head_vision.toml](qwen3_vl/vl/8b/w8a16_text_head_vision.toml) | [run.py](../run.py)<br>固定 P/D 复现 | `w8a16`；P=430；D=31；TP=8 | 继承 | `CONTROLLED_EVALUATION`<br>`QWEN3_VL_8B_RUNTIME_W8_CONTROLLED_EVALUATION_UNCERTIFIED` |

## rhinovla

[配置说明](rhinovla/README.md) · [权重准备](rhinovla/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [rhinovla/v1/fp16.toml](rhinovla/v1/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16` | 继承 | `IMPLEMENTED_UNVERIFIED_HW`<br>`RHINOVLA_OFFICIAL_NONRTC_UNVERIFIED` |
| [rhinovla/v3/controlled/fp16.toml](rhinovla/v3/controlled/fp16.toml) | [validate_rhinovla_v3.py](../../scripts/validate_rhinovla_v3.py) `--config`<br>独立验证片段 | `—` | 12（验证 CLI 默认，可覆盖） | 未登记；独立验证片段 |
| [rhinovla/v3/controlled/w8a16_action.toml](rhinovla/v3/controlled/w8a16_action.toml) | [validate_rhinovla_v3.py](../../scripts/validate_rhinovla_v3.py) `--config`<br>独立验证片段 | `—` | 12（验证 CLI 默认，可覆盖） | 未登记；独立验证片段 |
| [rhinovla/v3/controlled/w8a16_pipeline.toml](rhinovla/v3/controlled/w8a16_pipeline.toml) | [validate_rhinovla_v3.py](../../scripts/validate_rhinovla_v3.py) `--config`<br>独立验证片段 | `—` | 12（验证 CLI 默认，可覆盖） | 未登记；独立验证片段 |
| [rhinovla/v3/controlled/w8a16_pipeline_high.toml](rhinovla/v3/controlled/w8a16_pipeline_high.toml) | [validate_rhinovla_v3.py](../../scripts/validate_rhinovla_v3.py) `--config`<br>独立验证片段 | `—` | 12（验证 CLI 默认，可覆盖） | 未登记；独立验证片段 |
| [rhinovla/v3/controlled/w8a16_pipeline_high_fused.toml](rhinovla/v3/controlled/w8a16_pipeline_high_fused.toml) | [validate_rhinovla_v3.py](../../scripts/validate_rhinovla_v3.py) `--config`<br>独立验证片段 | `—` | 12（验证 CLI 默认，可覆盖） | 未登记；独立验证片段 |
| [rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_k.toml](rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_k.toml) | [validate_rhinovla_v3.py](../../scripts/validate_rhinovla_v3.py) `--config`<br>独立验证片段 | `—` | 12（验证 CLI 默认，可覆盖） | 未登记；独立验证片段 |
| [rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_qk.toml](rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_qk.toml) | [validate_rhinovla_v3.py](../../scripts/validate_rhinovla_v3.py) `--config`<br>独立验证片段 | `—` | 12（验证 CLI 默认，可覆盖） | 未登记；独立验证片段 |
| [rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_qk_partial_rope.toml](rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_qk_partial_rope.toml) | [validate_rhinovla_v3.py](../../scripts/validate_rhinovla_v3.py) `--config`<br>独立验证片段 | `—` | 12（验证 CLI 默认，可覆盖） | 未登记；独立验证片段 |
| [rhinovla/v3/controlled/w8a16_pipeline_input_cache.toml](rhinovla/v3/controlled/w8a16_pipeline_input_cache.toml) | [validate_rhinovla_v3.py](../../scripts/validate_rhinovla_v3.py) `--config`<br>独立验证片段 | `—` | 12（验证 CLI 默认，可覆盖） | 未登记；独立验证片段 |
| [rhinovla/v3/fp16.toml](rhinovla/v3/fp16.toml) | [run.py](../run.py)<br>调用者 factory | `fp16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/w8a16_action.toml](rhinovla/v3/w8a16_action.toml) | [run.py](../run.py)<br>调用者 factory | `w8a16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/w8a16_action_adarms.toml](rhinovla/v3/w8a16_action_adarms.toml) | [run.py](../run.py)<br>调用者 factory | `w8a16` | 继承 | 调用者 factory 模板 |
| [rhinovla/v3/w8a16_pipeline.toml](rhinovla/v3/w8a16_pipeline.toml) | [run.py](../run.py)<br>调用者 factory | `w8a16` | 12 | 调用者 factory 模板 |
| [rhinovla/v3/w8a16_pipeline_high_fused_vector_qk.toml](rhinovla/v3/w8a16_pipeline_high_fused_vector_qk.toml) | [run.py](../run.py)<br>调用者 factory | `w8a16` | 12 | 调用者 factory 模板 |

## wall_oss

[配置说明](wall_oss/README.md) · [权重准备](wall_oss/README.md#quantization)

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
| [wall_oss/legacy/fp16.toml](wall_oss/legacy/fp16.toml) | [run.py](../run.py)<br>历史复现 | `fp16` | 继承 | `SUPPORTED_LIMITED`<br>`WALL_OSS_RC9_FP16_DELIVERY_SUPPORTED` |
| [wall_oss/legacy/w4a16_channel.toml](wall_oss/legacy/w4a16_channel.toml) | [run.py](../run.py)<br>历史复现 | `w4a16` | 继承 | `EXPERIMENTAL`<br>`WALL_OSS_W4A16_EXPERIMENTAL_RETEST` |
| [wall_oss/legacy/w4a16_g32.toml](wall_oss/legacy/w4a16_g32.toml) | [run.py](../run.py)<br>历史复现 | `w4a16-pgrp` | 继承 | `EXPERIMENTAL`<br>`WALL_OSS_W4A16_PGRP_EXPERIMENTAL_RETEST` |
| [wall_oss/legacy/w8a16.toml](wall_oss/legacy/w8a16.toml) | [run.py](../run.py)<br>历史复现 | `w8a16` | 继承 | `EXPERIMENTAL`<br>`WALL_OSS_W8A16_EXPERIMENTAL_RETEST` |
| [wall_oss/rtc/fp16.toml](wall_oss/rtc/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16` | 继承 | `CONTROLLED_EVALUATION`<br>`WALL_OSS_RTC_FP16_CONTROLLED_TEMPORARY_SUPPORT` |
| [wall_oss/rtc/w8a16.toml](wall_oss/rtc/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16` | 继承 | `CONTROLLED_EVALUATION`<br>`WALL_OSS_RTC_W8A16_CONTROLLED_TEMPORARY_SUPPORT` |
| [wall_oss/sorting/fp16.toml](wall_oss/sorting/fp16.toml) | [run.py](../run.py)<br>运行模板 | `fp16` | 继承 | 未登记；模型专用模板 |
| [wall_oss/sorting/w8a16.toml](wall_oss/sorting/w8a16.toml) | [run.py](../run.py)<br>运行模板 | `w8a16` | 继承 | 未登记；模型专用模板 |
| [wall_oss/sorting/w8a16_action_w4a16_g32.toml](wall_oss/sorting/w8a16_action_w4a16_g32.toml) | [run.py](../run.py)<br>运行模板 | `w8a16_action_w4a16` | 继承 | 未登记；模型专用模板 |

## whisper_asr

[配置说明](whisper_asr/README.md) · [权重准备](whisper_asr/README.md#quantization)

| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |
|---|---|---|---|---|
| [whisper_asr/base/fp16.toml](whisper_asr/base/fp16.toml) | [run.py](../run.py)<br>运行模板 | `—` | 4 | 未登记；模型专用模板 |

合计 **294** 份配置。
