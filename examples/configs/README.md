# Public model configurations

每个模型只保留一套当前配置目录，按版本、组件、尺寸和精度区分。文件名中的不同精度、相机数、P/D 长度表示不同工作量，不是新旧格式的重复副本。

| 模型 | 唯一配置目录 | 内容 |
|---|---|---|
| Qwen3 | [qwen3/text](qwen3/README.md) | 尺寸、FP16/W8、W4/G32 与 LM-head recipe |
| Qwen3-VL | [qwen3_vl/{text,vision,vl}](qwen3_vl/README.md) | 独立 Text、Vision、图文及精度/长度组合 |
| Qwen3.5 | [qwen3_5/{text,vl,legacy}](qwen3_5/README.md) | Dense、MoE、精度、核数和长度；legacy 是独立 checkpoint 格式 |
| Pi0.5 | [pi05/{2cam,3cam}](pi05/README.md) | 8 核 FP16、W8、Action NVFP4、Prefill A8 + Action NVFP4 |
| LingBot | [lingbot/v2](lingbot/v2) | 6B FP16、W8A16；只保留 v2 配置 |
| RhinoVLA | [rhinovla/{v1,v3}](rhinovla/README.md) | 按版本选择调用方提供的 trusted runtime factory |
| Wall-OSS | [wall_oss/base](wall_oss/README.md) | 公开基础模型的四种精度配置 |
| DINOv3 | [dinov3/vit_b](dinov3/vit_b) | ViT-B FP16 |
| Gemma4 | [gemma4/e4b](gemma4/e4b) | E4B FP16 |
| Llama | [llama/3_2_1b](llama/3_2_1b) | Llama 3.2 1B FP16 |
| Hy-Embodied | [hy_embodied/umi](hy_embodied/umi) | 公开 UMI 配置 |
| HALO | [halo/action_expert](halo/action_expert) | Action expert 单步组件 |
| 其他集成入口 | [g05/base](g05/base)、[gr00t/n1_7_droid](gr00t/n1_7_droid)、[internvla/navdp](internvla/navdp)、[siglip/pi05_component](siglip/pi05_component) | 各自公开模型/组件合同 |

```sh
python examples/run_model.py --config examples/configs/qwen3_5/vl/2b/fp16.toml --check-config
python examples/run_model.py --config examples/configs/qwen3_5/vl/2b/fp16.toml
```

从仓库根目录运行。`--check-config` 不加载权重或访问板卡。alias 在 `RPU_MODEL_CACHE` 下解析，默认根目录为 `~/.cache/rhinoforge/models`；相对输入路径以进程工作目录为基准。

模型配置使用 `[example]`、`[input]`、`[run]` 和可选 `[rpu_execution]`，Pi 使用 `[pi05]`。`example.profile_id` 对应 [profiles.json](profiles.json) 中实际被模板引用的精度、组件和冷态约束。G0.5、GR00T、NavDP、SigLIP 的单个集成模板使用 `[runner/model/request]`；调用方自有的旧格式文件仍可运行，目录中不另放兼容副本。

短测请复制目标配置，将 `run.warmup=1`、`run.runs=2` 或 `3`，生成模型的 `input.decode_steps=32`，并使 `warmup_decode_steps` 不超过它。保留需要比较的其他输入、精度和执行设置。长 P/D 配置仅在显式选择时运行。

`assets/example.ppm` 是公开示意图片，不是历史性能或质量报告的原始输入。权重、batch、预处理 tensor 和应用请求由调用方提供。配置存在不扩大运行时准入或声明质量认证；本目录不包含客户模型、私有 fixture 或部署定制。
