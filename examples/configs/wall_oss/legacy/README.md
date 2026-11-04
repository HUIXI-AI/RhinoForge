<a id="wall-oss-原-registry-配置"></a>

# Wall-OSS registry 配置

本组通过 registry 别名加载公开 Wall-OSS 0.5，输入为三张图片、指令、状态和 mask。使用 `[example]` 格式，固定动作长度 32、十步去噪；按原始输入重放时选本组，按两/三图与分辨率选型见 [base](../base/README.md)。

<a id="选择配置"></a>

<a id="配置与权重对应"></a>

<a id="weight-recipes"></a>

## 配置说明

| 配置 | Vision | Language / Action decoder | 权重获取 |
|---|---|---|---|
| [fp16.toml](fp16.toml) | FP16 | FP16 | [原始公开模型](#download-parent) |
| [w8a16.toml](w8a16.toml) | W8 | W8 | [原始模型 + W8](#公开-base-legacy-w8a16) |
| [w4a16_channel.toml](w4a16_channel.toml) | W8 | 逐通道 INT4 | [原始/W8 + 逐通道 W4](#公开-base-legacy-w4逐通道-group0) |
| [w4a16_g32.toml](w4a16_g32.toml) | W8 | 分组 INT4 | [原始/W8 + G32 W4](#公开-base-legacy-w4-pgrp混合-group32) |

<a id="fp16-基准与配置增量"></a>

### FP16 基准

FP16 基准为 `fp16.toml`，主投影和设备 IO 为 FP16，CPU 预处理/归一化为 FP32；已使用分阶段 Graph 和融合去噪。模板不设置额外精度开关，执行参数由普通 Wall-OSS 路径选择。

### W8A16

`example.dtype` 从 `"fp16"` 改为 `"w8a16"`，同步替换 profile/registry 别名并增加浮点父别名。Vision block、Language expert0 和 Action expert1 的 Q/K/V/O/Gate/Up/Down 为逐输出通道 INT8，scale 和激活为 FP16。

patch/merger、norm/bias、embedding、动作预处理及输出回投保持浮点：设备部分 FP16，CPU 部分 FP32。转换器量化的额外二维浮点权重中，运行器只读取这些主体投影；其余模块仍从原始父目录取值。

### W4 G0：在 W8 配方上量化两组 decoder

`example.dtype="w4a16"`，别名改为 `wall-oss-0.5-w4a16`。Vision 继续读 W8；Language expert0 和 Action expert1 的 Q/K/V/O/Gate/Up/Down 改为逐输出通道 INT4、FP16 scale 和激活。离线使用 `--group-size 0`，加载时打包为运行所需的 pgrp 布局。

### W4 G32：两组 decoder 分组量化

`example.dtype="w4a16-pgrp"`，别名改为 `wall-oss-0.5-w4a16-pgrp`。离线使用 `--group-size 32`，沿 K 每 32 个权重计算一组 FP16 scale。Language expert0 和 Action expert1 的 Q/K/V/O/Gate/Up/Down 全部使用分组 INT4。

当前转换器生成的 G32 文件覆盖 Language 与 Action 的七类投影。旧 G32 包若仍把 Language expert0 Down 保存为 INT8，加载器默认通过 `RPU_WALL_OSS_EXPERT0_DOWN_INT4=1` 将该投影转换为逐通道 INT4；其余投影仍按包中的分组读取。

INT4 值保存在 INT8 容器中，加载时打包；容器类型不改变逻辑位宽。这两档 W4 同时量化 Language 和 Action，均使用整数格式；base 的动作 NVFP4 使用单独配方。

转换使用 [W8 CLI](../../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py)和 [W4 转换器](../../../../python/rpu_backend/quant/convert_wall_oss.py)，加载细节见 [llm.py](../../../../python/rpu_backend/adapters/wall_oss/llm.py)。

<a id="准备权重并运行"></a>

<a id="quantization"></a>

<a id="weight-sources"></a>

<a id="weights"></a>

## 准备与运行

### 获取所选权重

源权重为 X Square Robot 的 [wall-oss-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5/tree/b7dc23a82dbf70a65859c9fdd596703234597028)，revision `b7dc23a82dbf70a65859c9fdd596703234597028`，文件含 BF16 和 FP32。保留 processor/tokenizer 与 normalizer，源码见 [wall-x](https://github.com/X-Square-Robot/wall-x)。

下面统一以 `RPU_MODEL_CACHE` 定位。原始和转换目录放在同一缓存根下，便于解析 `source` 元数据。所有转换都直接读取原始目录；W4 仍需要 W8 目录提供 Vision。

<a id="download-parent"></a>

<a id="run-fp16"></a>

**原始目录：FP16 直接使用。**

```bash
export RPU_MODEL_CACHE=/absolute/path/model-cache
WALL_SRC="$RPU_MODEL_CACHE/wall-oss-0.5"
hf download x-square-robot/wall-oss-0.5 \
  --revision b7dc23a82dbf70a65859c9fdd596703234597028 \
  --local-dir "$WALL_SRC"
```

<a id="公开-base-legacy-w8a16"></a>

**W8：三种量化配置均需准备。**

```bash
python python/rpu_backend/quant/convert_wall_oss_w8a16.py \
  --src "$WALL_SRC" --dst "$RPU_MODEL_CACHE/wall-oss-0.5-w8a16"
```

<a id="公开-base-legacy-w4逐通道-group0"></a>

**逐通道 W4：选择逐通道模板时执行。**

```bash
python -m rpu_backend.quant.convert_wall_oss \
  --src "$WALL_SRC" \
  --dst "$RPU_MODEL_CACHE/wall-oss-0.5-w4a16" --group-size 0
```

<a id="公开-base-legacy-w4-pgrp混合-group32"></a>

**G32 W4：选择分组模板时执行。**

```bash
python -m rpu_backend.quant.convert_wall_oss \
  --src "$WALL_SRC" \
  --dst "$RPU_MODEL_CACHE/wall-oss-0.5-w4a16-pgrp" --group-size 32
```

`example.registry_alias` 与上面四个子目录同名。量化模板的 `input.fp16_registry_alias="wall-oss-0.5"` 绑定原始目录；保留两项别名即可，不添加 `input.checkpoint`。

<a id="使用"></a>

### 绑定输入并运行

```bash
WALL_CONFIG=examples/configs/wall_oss/legacy/w8a16.toml
cp "$WALL_CONFIG" /tmp/wall_registry.toml
# input.images/instruction/proprioception/state_mask = 配对实际输入。
# run.output_dir = 本次结果目录；保留 registry_alias 与 fp16_registry_alias。
vi /tmp/wall_registry.toml
python examples/run.py --config /tmp/wall_registry.toml
```

<a id="输入与运行参数"></a>

### 输入与常调参数

提供 face/left/right 三张图片、非空指令、26 维状态和 mask，实际图像 token 展开后的前缀范围为 `[545,640]`。动作长度 32，十步去噪。

| 字段 | 模板值 | 可选范围 | 修改时的关联 |
|---|---|---|---|
| `input.images/instruction` | 图片占位路径、默认指令 | 三张图片、非空文本 | 实际前缀保持在 545–640 |
| `input.proprioception/state_mask` | 26 维，mask 选择前 7 维 | 配对状态和维度选择 | mask 与输入状态及动作槽位一致 |
| `input.noise_seed` | 0 | 非负整数 | 固定输入噪声 |
| `run.warmup/runs` | 2 / 3 | 预热≥0、采样≥1 | 按所需运行次数调整 |
| `cpu.threads` | 未指定 | 正整数 | 省略时继承运行环境 |

公共字段见[参数说明](../../PARAMETERS.md)，运行库见[算子库说明](../../../../docs/runtime_assets.md)。更换冷配置后使用新进程。
