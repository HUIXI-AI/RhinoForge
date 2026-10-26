# Pi0.5 配置

本目录使用 LeRobot 的 LIBERO 微调模型。先按相机槽位选择 `2cam/` 或 `3cam/`，再选择精度；两组使用同一套权重配方。输入为 batch1，动作长度 50，十步去噪。

<a id="选择配置"></a>

<a id="配置与权重对应"></a>

<a id="weight-recipes"></a>

## 配置说明

| 配方 | 两图 TP8 | 三图 TP8 | 使用的权重与获取步骤 |
|---|---|---|---|
| FP16 | [配置](2cam/fp16.toml) | [配置](3cam/fp16.toml) | [原始 LIBERO 目录](#download-parent) |
| W8A16 | [配置](2cam/w8a16.toml) | [T32 配置](3cam/w8a16.toml) / [T160 配置](3cam/w8a16_t160.toml) | [W8 转换目录](#run-w8) |
| W8 + 动作 NVFP4 | [配置](2cam/w8a16_action_nvfp4.toml) | [T32 配置](3cam/w8a16_action_nvfp4.toml) / [T160 配置](3cam/w8a16_action_nvfp4_t160.toml) | [动作 NVFP4 转换目录](#w8--action-nvfp4离线转换后运行) |
| 预填充 W8A8 + 动作 NVFP4 | [配置](2cam/w8a16_prefill_w8a8_action_nvfp4.toml) | [T32 配置](3cam/w8a16_prefill_w8a8_action_nvfp4.toml) / [T160 配置](3cam/w8a16_prefill_w8a8_action_nvfp4_t160.toml) | [复用动作 NVFP4 目录](#w8--prefill-w8a8--action-nvfp4) |

三图 T32 还提供 [FP16 TP4](generic/3cam/fp16_tp4.toml) / [TP6](generic/3cam/fp16_tp6.toml) 和 [W8 TP4](generic/3cam/w8a16_tp4.toml) / [TP6](generic/3cam/w8a16_tp6.toml)，分别复用上表原始目录和 W8 目录。全部 12 份配置对应三套权重，无需按相机数或核数重新转换。

<a id="fp16-基准与配置增量"></a>

### FP16 基准

以同相机数的 `fp16.toml` 为基准。TP8 默认配置已经启用多相机 SigLIP 合批、Vision/Prefill/去噪 Graph replay、预填充分成两段执行、KV 直接缓存、十步去噪展开及融合。AdaRMS 条件参数在 Graph 准备时预计算，推理时读取参数表；FP16 的 SigLIP 和 AdaRMS dense 均保持浮点。默认 Linear 使用 ACC16，指累加精度，与权重位宽分别选择。

### W8A16：基准上量化三个组件

`pi05.precision` 从 `"fp16"` 改为 `"w8a16"`，`input.checkpoint` 改为配对 W8 目录。实际范围为：

- SigLIP 的 Q/K/V/O、FC1/FC2。
- VLM 和 Action 的 Q/K/V/O、Gate/Up/Down。
- AdaRMS 的 dense 条件投影；这些投影用于准备预计算表。

W8A16 使用逐输出通道 INT8 权重、FP16 scale 和 FP16 激活。patch embedding、位置、归一化参数和偏置、多模态投影、动作输入/输出投影保持 FP16。离线转换器只量化 VLM/Action；SigLIP 和 AdaRMS 是加载阶段的增量，不能仅根据权重文件中的 dtype 判断运行精度。

### 动作 NVFP4：在 W8 配方上改变动作五类投影

`pi05.precision` 改为 `"w8a16_action_nvfp4"`，切换到动作 NVFP4 离线目录。Action 的 Q/O/Gate/Up/Down 使用 E2M1 四位浮点权重、每 16 个 K 维元素一组的 E4M3FN scale 和 FP32 tensor scale；K/V 继续使用 W8，激活保持 FP16。

该模板同时选择 NVFP4 执行路径。TP8、T32/T160、Action ACC16 时启用对应的 Gate/Up/GeGLU 融合；Action ACC32 或其他文本长度使用所选精度的分开投影路径。FP16/W8 的 GeGLU 融合已有各自实现。

### 预填充 W8A8：在动作 NVFP4 上增加一处激活量化

`pi05.precision` 改为 `"w8a16_prefill_w8a8_action_nvfp4"`，权重目录不变。唯一新增的量化范围是 VLM 预填充 Gate/Up 的输入激活：由 FP16 转为 INT8，与原有 W8 权重组成 W8A8；其余投影仍为 A16。激活随输入在推理时转换，因此没有新的离线权重包。

以上执行设置由 [runner](../../runners/pi05/optimized.py) 选择；权重处理见[离线转换器](../../../python/rpu_backend/quant/convert_pi05.py)、[Pi0.5 adapter](../../../python/rpu_backend/adapters/pi05/__init__.py)和 [SigLIP adapter](../../../python/rpu_backend/adapters/siglip.py)。

### 相机数与核数

模板默认 T32，TP8 使用多相机合批和两段预填充。T32 时两图 chunk 为 272，三图为 400。`generic/3cam/` 的 TP4/6 配置使用逐相机 Vision 和通用执行路径，AdaRMS 不采用 TP8 的预计算表；这是核数和执行路径的共同变化。

<a id="准备权重并运行"></a>

<a id="quantization"></a>

<a id="weight-sources"></a>

<a id="weights"></a>

## 准备与运行

### 获取所选权重

源权重为 [lerobot/pi05_libero_finetuned_v044](https://huggingface.co/lerobot/pi05_libero_finetuned_v044/tree/dbf8a3f794a9c4297b44f40b752712f50073d945)，revision 为 `dbf8a3f794a9c4297b44f40b752712f50073d945`。发布文件含 BF16 和 FP32；运行精度由所选配置决定。`model.safetensors` 的 LFS SHA256 为 `877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae`。保留目录中的配置和预处理资产。[pi05_base](https://huggingface.co/lerobot/pi05_base) 是另一套权重。

以下命令在仓库根目录执行。先下载原始模型，再按所选配方执行一次转换；W8 和 NVFP4 都从原始目录分别生成。

<a id="download-parent"></a>

<a id="run-fp16"></a>

**原始目录：FP16 直接使用。**

```bash
PI05_SRC=/absolute/path/pi05_libero_finetuned
hf download lerobot/pi05_libero_finetuned_v044 \
  --revision dbf8a3f794a9c4297b44f40b752712f50073d945 \
  --local-dir "$PI05_SRC"
```

<a id="run-w8"></a>

**W8 目录：从原始模型转换。**

```bash
PI05_W8=/absolute/path/pi05_w8a16
python python/rpu_backend/quant/convert_pi05.py \
  --src "$PI05_SRC" --dst "$PI05_W8"
```

<a id="w8--action-nvfp4离线转换后运行"></a>

<a id="w8--prefill-w8a8--action-nvfp4"></a>

**动作 NVFP4 目录：两种 NVFP4 配置共用。**

```bash
PI05_NVFP4=/absolute/path/pi05_action_nvfp4
python python/rpu_backend/quant/convert_pi05.py \
  --src "$PI05_SRC" --dst "$PI05_NVFP4" \
  --action-w4 --action-w4-format nvfp4
```

启用预填充 W8A8 只需选择相应 TOML，继续使用 `PI05_NVFP4`。
官方 checkpoint 可省略与 embedding 绑定且推理路径不调用的两个 `lm_head.weight`；loader
会按该结构禁用对应 head，不需要复制或补造约 1.58 GB 的无用权重。

<a id="使用"></a>

### 绑定输入并运行

从第一章选择 TOML。下面以三图 W8 为例，其他配置只改 `PI05_CONFIG` 并按表绑定 checkpoint：

```bash
PI05_CONFIG=examples/configs/pi05/3cam/w8a16.toml
cp "$PI05_CONFIG" /tmp/pi05.toml
# input.checkpoint：FP16 填 PI05_SRC 的实际路径，W8 填 PI05_W8，NVFP4 两档填 PI05_NVFP4。
# input.batch：匹配相机槽位和文本长度的 CPU batch 文件。
# run.output_dir：本次结果目录。
vi /tmp/pi05.toml
python examples/run.py --config /tmp/pi05.toml
```

TOML 中填写实际绝对路径。保留模板 registry 别名时，`pi05-libero-finetuned` 和 `pi05-libero-finetuned-w8a16-vlm-expert` 分别由 `RPU_MODEL_CACHE` 定位原始和 W8 目录。

<a id="输入与输出"></a>

### 输入格式

`input.batch` 为 CPU tensor 字典。每个 `observation.images.*` 槽位形状为 `[1,3,256,256]`，与 checkpoint 配置对应。两图提供两路真实相机；三槽提供配置中的全部槽位，可包含其定义的 `empty_camera_*` 占位槽。语言 token 为整数 `[1,T]`，attention mask 为同宽、右 padding 的非空 bool；入口直接使用实际 token。图像 padding mask 若存在，应为 `[1]` bool true。其余字段遵循 `Pi05Policy.predict_action_chunk`。

<a id="输入与执行变体"></a>

### 常调参数

| 字段 | 模板值或入口默认 | 可选范围 | 修改时的关联 |
|---|---|---|---|
| `input.cameras` | 2 或 3 | 2 / 3 | 换成匹配的 batch 槽位；同精度权重复用 |
| `input.text_tokens` | 32 | TP8：32/64/96/128/160；TP4/6：32 | token 和 mask 同宽；TP8 chunk 为 `(256×相机数+T)/2` |
| `rpu_execution.model.num_cores` | 未写时 8；缩核模板为 4/6 | 4 / 6 / 8 | TP4/6 仅三图 T32 FP16/W8；NVFP4/A8 使用 TP8 |
| `rpu_execution.prefill.chunk_size` | auto | 自动或有效整数 | NVFP4/A8 使用上述两段预填充尺寸 |
| `rpu_execution.prefill/vision/action.linear_acc32` | 未写时 false | false / true | 分组件选择 ACC16/ACC32，权重目录不变 |
| `pi05.instruction_residency` | 模板省略 | auto / off | 按运行条件选择快照重放；off 保留普通 Graph replay |
| `cpu.threads` | T160 NVFP4 模板为 4；其他模板省略、入口默认 8 | 正整数 | CPU 线程，与设备核数分别设置；限制同机应用争抢 |
| `run.warmup/runs/seed` | 2 / 6 / 0 | 预热≥0、采样≥1、seed≥0 | `input.noise` 可指定固定 CPU 噪声文件 |

例如 T64/96/128/160 时，两图 chunk 为 288/304/320/336，三图为 416/432/448/464。公共运行、诊断和输出字段见[参数说明](../PARAMETERS.md)；运行库采用[正式 main/DDR 两份 REF](../../../docs/runtime_assets.md)。更换权重、核数或 ACC 后使用新进程。

## 输出

输出 `report.json` 和 `actions.pt`，分别保存有效配置与动作输出。
