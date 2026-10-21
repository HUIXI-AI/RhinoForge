# 按模型选择配置

[示例入口](../README.md) · [全部 TOML](CONFIGURATIONS.md) · [权重准备](WEIGHTS.md) · [通用参数](PARAMETERS.md) · [命名规则](NAMING.md)

模型页列出配置差异、下载或转换步骤、输入格式和运行命令。

| 模型族 | 配置说明 | 主要区别 |
|---|---|---|
| G0.5 | [g05](g05/README.md) | 2/3cam × FP16、W8、W8+Action NVFP4；ACC/text 在文件内修改 |
| Pi0.5 | [pi05](pi05/README.md) | 2/3cam × 四精度；TP8 优化与 TP4/6 通用执行 |
| HyVLA | [hy_vla](hy_vla/README.md) | 2/3cam × 三精度；当前优化入口为 TP8 |
| Wall-OSS | [wall_oss](wall_oss/README.md) | base 按相机/分辨率/精度；RTC 与分拣为不同负载 |
| GR00T | [gr00t](gr00t/README.md) | 224/256 × FP16/W8，text 在文件内修改 |
| RhinoVLA | [rhinovla](rhinovla/README.md) | v1 官方/调用者 factory、v3 factory 与九种独立配置 |
| Qwen3-ASR | [qwen3_asr](qwen3_asr/README.md) | 0.6B 音频转文字，固定 FP16 配方 |
| Whisper ASR | [whisper_asr](whisper_asr/README.md) | base 离线转写，自动语言、固定 FP16 配方 |
| Qwen3-TTS | [qwen3_tts](qwen3_tts/README.md) | 0.6B Base；默认卸载 12 层残差卷积到 RPU，可显式切回 CPU |
| Qwen3 | [qwen3](qwen3/README.md) | 模型大小、精度、Instruct / lm_head recipe |
| Qwen3.5 | [qwen3_5](qwen3_5/README.md) | text/VL、Dense/MoE、精度、固定 P/D 输入 |
| Qwen3-VL | [qwen3_vl](qwen3_vl/README.md) | text/VL、模型大小、精度、固定 P/D 输入 |
| DINOv3 | [dinov3](dinov3/README.md) | ViT-B FP16 图像编码 |
| Gemma4 | [gemma4](gemma4/README.md) | E4B FP16 |
| Llama | [llama](llama/README.md) | 3.2-1B FP16 |
| HY Embodied | [hy_embodied](hy_embodied/README.md) | 三图 UMI FP16 接口；其他配方见 HyVLA |
| HALO | [halo](halo/README.md) | Action expert；组件配置 |
| InternVLA | [internvla](internvla/README.md) | System2/RGBD/NavDP 完整链，混合精度 |
| LingBot | [lingbot](lingbot/README.md) | v1 CPU/设备混合；v2 FP16/W8 与 exact-Z2 |

## 如何选择

先确定模型版本和任务，再选择输入规格与精度配方。`text/`、`vl/`、`vision/` 表示执行组件；相机槽位、分辨率和模型大小由目录区分。同一组权重可以被不同输入和核数的配置复用，具体关系写在模型页。

FP16、W8、W4 等文件名描述运行配方；实际量化范围由名称中的模块限定和模型页共同说明。固定 P/D、TP、ACC 变体的含义见[命名规则](NAMING.md)，所有路径见[配置索引](CONFIGURATIONS.md)。

公共运行入口的 TOML 为完整模板；RhinoVLA `controlled/` 使用对应页面的独立命令。改变精度配方时直接复制对应文件，再按其获取步骤准备权重；输入路径、线程和采样次数等常调项按模型页填写。

<a id="配置日志输出"></a>

日志级别与产物说明见[日志与结果](../RESULTS.md#日志设置)，公共字段见[通用参数](PARAMETERS.md)。

<a id="本轮路径迁移"></a>

文件改名记录见[路径对照](config_names.json)；命令行使用当前配置路径。
