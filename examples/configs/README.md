# 配置总览

选择模型后，在同一页查看全部配置、默认值、可调输入范围和联动限制。

[示例入口](../README.md) · [权重准备](WEIGHTS.md) · [通用参数](PARAMETERS.md) · [核数支持矩阵](../../docs/planner_core_topology.zh.md) · 目录维护

| 模型 | 变体 | 尺寸 | 配置数 |
|---|---|---|---:|
| [DINOv3](dinov3/README.md#configurations) | ViT-B | 见模型说明 | 1 |
| [G0.5](g05/README.md#configurations) | 2cam、3cam、Legacy | — | 7 |
| [Gemma4](gemma4/README.md#configurations) | E4B | 见模型说明 | 1 |
| [GR00T](gr00t/README.md#configurations) | N1.7 DROID | 见模型说明 | 8 |
| [HALO](halo/README.md#configurations) | Action Expert | 见模型说明 | 1 |
| [HY Embodied](hy_embodied/README.md#configurations) | UMI | 见模型说明 | 1 |
| [HyVLA](hy_vla/README.md#configurations) | 2cam、3cam | 见模型说明 | 6 |
| [InternVLA](internvla/README.md#configurations) | NavDP | 见模型说明 | 1 |
| [LingBot](lingbot/README.md#configurations) | Legacy、v1、v2 | 4B、6B | 6 |
| [Llama](llama/README.md#configurations) | 3.2-1B | 见模型说明 | 1 |
| [Pi0.5](pi05/README.md#configurations) | 2cam、3cam | 见模型说明 | 16 |
| [Qwen3](qwen3/README.md#configurations) | Text、Instruct | 0.6B、1.7B、4B、8B、14B、32B | 25 |
| [Qwen3.5](qwen3_5/README.md#configurations) | Text、MoE、VL | 0.8B、2B、4B、9B、27B、35B-A3B | 43 |
| [Qwen3 ASR](qwen3_asr/README.md) | Base | 0.6B、1.7B | 8 |
| [Qwen3 TTS](qwen3_tts/README.md) | Base | 0.6B、1.7B | 9 |
| [Qwen3-VL](qwen3_vl/README.md#configurations) | Text、Vision、Multimodal | 2B、4B、8B、32B | 32 |
| [RhinoVLA](rhinovla/README.md#configurations) | v1、v3 | 见模型说明 | 15 |
| [Wall-OSS](wall_oss/README.md#configurations) | base、Legacy、rtc、sorting | 见模型说明 | 21 |

输入范围来自配置校验和运行器约束，4/6/8 核支持与板上记录另见核数支持矩阵；默认值不代表范围上限。

本页由 `python scripts/check_examples.py --write` 生成。完整公开配置索引见 [CONFIGURATIONS.md](CONFIGURATIONS.md)。
