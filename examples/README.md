# 模型示例 / Model examples

配置统一按模型归档，每种版本、组件、精度和工作量只保留一份当前模板。完整目录见[配置索引](configs/README.md)。

```sh
python examples/run_model.py --config examples/configs/qwen3_5/vl/2b/fp16.toml --check-config
python examples/run_model.py --config examples/configs/qwen3_vl/vl/4b/w8a16.toml
```

`--check-config` 只检查配置，不加载模型或访问板卡。运行前安装对应依赖并配置运行时资产。TOML 中的模型、图片、batch 和输出路径相对启动命令的工作目录解析；`/path/to/...` 需要替换成调用方本地文件。配置不包含权重或私有输入。

| 模型 | 配置组织 |
|---|---|
| [Qwen3](configs/qwen3/README.md) | `qwen3/text/<尺寸>/<精度>.toml` |
| [Qwen3-VL](configs/qwen3_vl/README.md) | `qwen3_vl/{text,vision,vl}/<尺寸>/` |
| [Qwen3.5](configs/qwen3_5/README.md) | `qwen3_5/{text,vl,legacy}/<尺寸>/` |
| [Pi0.5](configs/pi05/README.md) | `pi05/{2cam,3cam}/`，仅 8 核配置 |
| LingBot | `lingbot/v2/6b_<精度>.toml`，仅 v2 |
| [RhinoVLA](configs/rhinovla/README.md) | `rhinovla/{v1,v3}/` |
| [Wall-OSS](configs/wall_oss/README.md) | `wall_oss/base/`，公开基础 checkpoint |
| 其他 | DINOv3、Gemma4、Llama、Hy-Embodied、HALO 等均在各自模型目录下 |

同一个配置可交给统一入口或对应模型脚本，例如 `examples/qwen3_5_vision.py --config ...`。Qwen 和策略示例使用 `[example/input/run]`，Pi 使用 `[pi05/input/run]`。少数独立集成入口使用 `[runner/model/request]`；不要混用字段，详见[运行时配置](../docs/runtime_config.zh.md)。

短测可复制目标配置，将 `[run]` 的 `warmup` 设为 1、`runs` 设为 2 或 3；生成模型把 `[input].decode_steps` 设为 32，确保 `warmup_decode_steps` 不超过它。保留原精度、核数、prefill、图像及 policy 步数，记录实际输入和计时模式。`decode_steps` 是 prefill 首 token 之后的 decode 调用数，EOS 可提前终止。D4096 文件不会自动执行。

`[run].profile = true` 或 `--profile` 会在计时后单独采集一次 Torch trace，结果写到 `run.output_dir` 下的独立目录，包含配置、样本时延和输出。`hwperf=true` 不由公开示例导出；结果不代表数值或任务质量认证。普通修改的检查预算见[验证说明](../docs/validation_policy.zh.md)。

## Qwen3-VL 精度

FP16、runtime W8、Text AWQ 与 runtime W4 的 checkpoint 格式不同，各尺寸必须使用自己的转换产物：

```sh
python -m rpu_backend.quant.convert_qwen3_vl --bits 8 --src /path/to/public-fp16 --dst /path/to/runtime-w8
python -m rpu_backend.quant.convert_qwen3_vl --bits 4 --src /path/to/public-awq --dst /path/to/awq-runtime-w4
```

AWQ 源须符合 symmetric G32 compressed-tensors 格式。FP16 源经 `--bits 4` 得到 RTN W4，不能作为 AWQ。各模板的 projection recipe 和量化元数据由加载器检查，见[Qwen3-VL 配置说明](configs/qwen3_vl/README.md)。

## Pi0.5 精度与输入

当前模板均为 B1、TP8、T32、动作长度 50、10 个 denoise step，通过公开 `optimized_profile` 选择融合、缓存和 AdaRMS 预计算。两相机、三相机各提供 FP16、W8、Action NVFP4、Prefill A8 + Action NVFP4；真实相机槽位和输入预处理必须匹配 checkpoint。

NVFP4 需要独立离线 checkpoint，使用 `striped_v2`、block16，Action Q/O/Gate/Up/Down 为 NVFP4，K/V 和 VLM 为 W8。Prefill A8 只改变指定 GateUp/GeGLU 路径。不得将旧 INT4 checkpoint 改名当作 NVFP4，详见[Pi 配置说明](configs/pi05/README.md)。

## 公开 Wall-OSS 与 RhinoVLA

Wall-OSS 使用公开 [wall-oss-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5) 底座的 processor、normalizer、state bins 和动作定义；配置集中在 `base/`。RhinoVLA 按 v1、v3 选择调用方信任的 factory 与请求输入；版本目录不代表内置了模型仓库的私有装配代码。
