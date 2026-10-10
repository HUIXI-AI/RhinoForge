# 模型入口与运行范围

**简体中文** | [English](model_support.md)

从[模型示例](../examples/README.zh.md)选择 TOML，按模型指南准备权重和输入，执行 `python -m examples.run --config PATH --check-config` 后运行。检查命令只验证配置。

支持范围按 checkpoint、精度、核数、输入形状和长度确定，由加载器与运行时准入检查执行。受控入口还需模板声明的 opt-in；详细参数见[配置索引](../examples/configs/CONFIGURATIONS.md)。

权重与请求数据由调用方提供。安装依赖见[模型资产](model_assets.zh.md)和[运行时资产](runtime_assets.zh.md)；调用方式见[运行与调试](model_testing.zh.md)、[API](api_reference.zh.md)和[量化](quantization.zh.md)。

## 模型与输入

下表链接到可执行配置及输入约定。模型系列名不能替代 checkpoint、精度和形状要求。

| 模型 | 入口与指南 | 输入 |
|---|---|---|
| Qwen3 / Llama | `RPUModelForCausalLM`；[Qwen3](../examples/configs/qwen3/README.md)、[Llama](../examples/configs/llama/README.md) | Token 与匹配的 KV cache |
| Qwen3-VL | `RPUModelForConditionalGeneration`；[指南](../examples/configs/qwen3_vl/README.md) | 文本、图片及 processor 生成的位置和网格信息 |
| LightNav-0 | `RPUModelForConditionalGeneration`；[R1 指南](lightnav_0_r1.zh.md) | 文本与视频历史；上游源码和 checkpoint 另行提供 |
| Qwen3.5 Dense / MoE | `RPUModelForConditionalGeneration`；[指南](../examples/configs/qwen3_5/README.md) | 文本或图文、对应 Dense/MoE cache 与 checkpoint 配置 |
| Gemma4 | `Gemma4Adapter`；[指南](../examples/configs/gemma4/README.md) | E4B Instruct 文本生成；CPU 前后处理与设备 decoder |
| DINOv3 | `DINOv3Adapter`；[指南](../examples/configs/dinov3/README.md) | 预处理后的图片 |
| Pi0.5 | `Pi05Policy`；[指南](../examples/configs/pi05/README.md) | 与 checkpoint 匹配的观测，可显式指定 action noise |
| Wall-OSS | `WallOssPolicy`；[基础](../examples/configs/wall_oss/README.md)、[RTC](../examples/configs/wall_oss/rtc/README.md)、[分拣](../examples/configs/wall_oss/sorting/README.md) | 图片、指令、本体状态及对应 mask |
| HY Embodied / HyVLA | `HyEmbodiedPolicy` 或组件 adapter；[UMI](../examples/configs/hy_embodied/README.md)、[HyVLA](../examples/configs/hy_vla/README.md) | 预处理后的相机、文本和状态张量 |
| GR00T | 专用 adapter/runner；[指南](../examples/configs/gr00t/README.md) | 匹配 checkpoint 的观测 batch 和 backbone 资产 |
| G0.5 | `patch_g05_policy_for_rpu`；[指南](../examples/configs/g05/README.md) | 相机、文本、状态输入包及外部 policy 源码 |
| InternVLA / NavDP | 完整链 runner 与组件 adapter；[指南](../examples/configs/internvla/README.md) | System2 → RGBD → NavDP 的配对权重包、CPU fixture 和资产清单 |
| LingBot v1 / v2 | 组件 adapter 或 `build_lingbot_vla_v2`；[v1](../examples/configs/lingbot/README.md)、[v2](../examples/configs/lingbot/v2/README.md) | v1 为预处理观测；v2 为权重、训练 YAML 及相机/状态/语言张量 |
| RhinoVLA | `RhinoVLAPolicy`；[指南](../examples/configs/rhinovla/README.md) | 模型仓库预处理及 runtime factory |
| HALO | Action Expert runner；[指南](../examples/configs/halo/README.md) | 配对 EMA 权重、单步 CPU fixture 和已准备的前缀 K/V |

核数、存储补齐与 cache 创建见[计算核拓扑](planner_core_topology.zh.md)，
adapter 所有权与共享运行约定见[实现索引](../knowledge/synthesis/model-family-map.md)。

## Qwen3 语音

`Qwen3ASRModel` 和 `Qwen3TTSModel` 提供独立的 ASR/TTS 入口，由 CPU 和设备组件
共同执行。固定资产、输入范围、源码准备和进程所有权见[语音指南](qwen3_audio.zh.md)。
使用 [ASR](../examples/configs/qwen3_asr/README.md) 和
[TTS](../examples/configs/qwen3_tts/README.md) TOML，通过 `examples.run` 调用语音 API；
仍不经过 CausalLM/VLM registry。
