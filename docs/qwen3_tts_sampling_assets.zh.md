# Qwen3-TTS RPU 采样资产使用说明

本文说明如何部署 Qwen3-TTS 的 RPU Predictor/Talker 采样资产，并通过
`Qwen3TTSModel` 运行一次可验证的语音合成。适用范围如下：

- 模型：`Qwen3-TTS-12Hz-0.6B-Base`；
- Rhino runtime：v1.2.0；
- RhinoForge：包含统一 TTS 采样资产加载支持的匹配源码或 wheel；
- 计算精度：模型 FP16，设备 Linear 按对应执行路径使用 ACC16/ACC32。

采样资产与 RhinoForge 代码必须配套交付。旧版 6 文件资产不能与新版代码混用。

## 1. 准备两个资产文件

将以下两个文件放入同一个只读目录，例如 `/opt/rhinoforge/tts-sampling`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `tts_sampling_unified.ref` | 26,392 B | `cc3e096e5684784f72cad83ff025ab00ec034a3b9565b11de20b70af94df116f` |
| `tts_sampling_tables.bin` | 22,064 B | `822492f65ab138b3e99d35a85565394df92e45d09afd27589bebbdf9bab3b928` |

部署后先检查文件和摘要：

```bash
export RPU_TTS_SAMPLING_ASSET_DIR=/opt/rhinoforge/tts-sampling

test -r "$RPU_TTS_SAMPLING_ASSET_DIR/tts_sampling_unified.ref"
test -r "$RPU_TTS_SAMPLING_ASSET_DIR/tts_sampling_tables.bin"
sha256sum \
  "$RPU_TTS_SAMPLING_ASSET_DIR/tts_sampling_unified.ref" \
  "$RPU_TTS_SAMPLING_ASSET_DIR/tts_sampling_tables.bin"
```

`tts_sampling_unified.ref` 包含 Predictor 与 Talker 共用的去重算子；
`tts_sampling_tables.bin` 通过版本头保存两条链各自的寄存器参数。加载器会拒绝文件
缺失、截断、尺寸或结构版本不匹配的情况，不会静默使用错误参数。

## 2. 设置运行环境

必须在启动 Python、加载模型之前设置：

```bash
export RPU_TTS_PREDICTOR_GROUP=1
export RPU_TTS_SAMPLING_EXPERIMENT="$RPU_TTS_SAMPLING_ASSET_DIR"
export RPU_TTS_TALKER_SAMPLING_EXPERIMENT="$RPU_TTS_SAMPLING_ASSET_DIR"
```

两个目录变量必须指向同一个两文件目录。模型加载后不要修改这些变量；如需切换资产，
关闭模型并启动新进程。

runtime 主算子库仍按统一安装指南配置，例如：

```bash
export RHINO_LAUNCH_ROOT=/path/to/rhino-launch-kernel-v1.2.0-linux-aarch64
export RPU_KERNEL_LIB_PATH=/path/to/RhinoForge-runtime-v1.2.0/rhinoOpLib_current.ref
export RPU_KERNEL_LIB_PATH_EXPANSION=/path/to/RhinoForge-runtime-v1.2.0/rhinoExpansionOpLib_current.ref
export LD_LIBRARY_PATH="$RHINO_LAUNCH_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

不要将 `tts_sampling_unified.ref` 配置到 `RPU_KERNEL_LIB_PATH` 或
`RPU_KERNEL_LIB_PATH_EXPANSION`。它是模型专用采样资产，不是 runtime 主算子库。

客户机器同时运行其他应用时，建议从 4 个 Torch CPU 线程开始，并在导入 Torch 前设置：

```bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
```

线程数会影响竞争场景中的 CPU 占用和 RTF，应在客户真实并行负载下复测。

## 3. 运行模型

准备以下本地路径：

```bash
export QWEN3_TTS_CHECKPOINT=/path/to/Qwen3-TTS-12Hz-0.6B-Base
export QWEN3_TTS_PREPARED=/path/to/qwen3-tts-prepared/qwen_tts
export QWEN3_TTS_REFERENCE=/path/to/mono-24k.wav
export QWEN3_TTS_OUTPUT=/tmp/qwen3-tts-rpu.wav
```

参考音频必须是单声道 24 kHz、0.5–20 秒、非静音音频。运行下面的 Python 程序：

```python
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
from rpu_backend.api import Qwen3TTSModel

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

with Qwen3TTSModel.from_pretrained(
    os.environ["QWEN3_TTS_CHECKPOINT"],
    source=os.environ["QWEN3_TTS_PREPARED"],
    dtype=torch.float16,
    codec_backend="rpu",
) as model:
    torch.manual_seed(0)
    started = time.perf_counter()
    waveform, sample_rate = model.synthesize(
        "这是 Qwen3-TTS RPU 采样资产验证。",
        reference_audio=os.environ["QWEN3_TTS_REFERENCE"],
        language="Chinese",
        max_new_tokens=256,
    )
    wall_seconds = time.perf_counter() - started
    stats = model.execution_stats()

assert sample_rate == 24000
assert waveform.ndim == 1 and waveform.size > 0
assert np.isfinite(waveform).all() and np.any(waveform)
assert all(part["graph_invariant_ok"] for part in stats["components"].values())
assert stats["components"]["predictor"]["group_hw_submits_total"] > 0

sf.write(os.environ["QWEN3_TTS_OUTPUT"], waveform, sample_rate,
         format="WAV", subtype="FLOAT")
print(json.dumps({
    "output": os.environ["QWEN3_TTS_OUTPUT"],
    "sample_rate": sample_rate,
    "audio_seconds": waveform.size / sample_rate,
    "wall_seconds": wall_seconds,
    "rtf": wall_seconds / (waveform.size / sample_rate),
    "generation": stats["generation"],
    "predictor_group_hw_submits_total":
        stats["components"]["predictor"]["group_hw_submits_total"],
}, ensure_ascii=False))
```

运行成功后应生成 24 kHz FLOAT WAV，并打印大于 0 的
`predictor_group_hw_submits_total`。`graph_invariant_ok=true` 只能证明 Graph 生命周期
正常；正式验收还需检查语音内容、音质、RTF 和竞争负载下的 CPU 使用情况。

## 4. 常见错误

| 现象 | 处理方式 |
|---|---|
| 报告 `TTS sampling table size/schema` | 检查 `tts_sampling_tables.bin` 的大小和 SHA256，不要使用旧参数文件拼接替代 |
| `.ref` 中找不到采样算子 | 检查 `tts_sampling_unified.ref` 的 SHA256，并确认 RhinoForge 与资产来自同一交付版本 |
| 报告配置已变化、要求 reload | 环境变量在模型加载后被修改；关闭模型并启动新进程 |
| 仍看到 CPU 采样或 RTF 明显升高 | 确认三个 `RPU_TTS_*` 变量在 Python 启动前已导出，且没有使用缺少统一资产加载支持的旧 wheel |
| RPU 设备忙 | 停止占用 `/dev/rpu` 的其他推理服务，取得平台设备锁后再运行；测试完成后恢复业务服务 |

完整模型准备、输入范围和流式接口见 [Qwen3 ASR / TTS 使用指南](qwen3_audio.zh.md)。
