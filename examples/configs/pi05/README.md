# Pi0.5 配置

[配置总览](../README.md) · [全部配置](#configurations) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

按 batch 的相机槽数量选择 `2cam/` 或 `3cam/`，按 action horizon 选择其下的 `h50/` 或 `h10/`（两档使用不同权重），再选择 FP16、W8A16、Action NVFP4 或 Prefill W8A8 + Action NVFP4 配方。两档的配方一一对应，例如 `2cam/h50/fp16.toml` 与 `2cam/h10/fp16.toml`。Leju Kuavo 的 openpi pi05 原始观测（三路相机原尺寸帧 + 指令 + 物理 state）使用 `kuavo/`（真机 848×480）或 `kuavo/sim/`（仿真 640×480），见 [Kuavo 原始观测](#kuavo)。

<a id="配置矩阵"></a>

## 输入与核心

`input.text_tokens` 是 token tensor 的存储宽度，可选 **32、64、96、128、160、192**，默认 **32**；192 只用于 3 相机的 FP16 / W8A16。实际有效 token 数由 mask 决定，两者分别核对；mask 为 False 的 pad 行在 prefill 前置零，不会在 FP16 中溢出（见[常见坑](../../../docs/pitfalls.md#other-high-value-checks)）。batch 的相机输入为每路 `[1,3,256,256]`，token 和 mask 为 `[1,T]`；修改相机数或 T 后，batch 也要匹配。`kuavo/` 的输入是相机原尺寸帧（真机 848×480、仿真 640×480），每次推理在 host 端按 openpi 缩放到 `[1,3,224,224]`，与 pi05_base 声明的图像特征一致。

默认 8 核。2/3 相机的四种精度配方都允许 T32–T160；4/6 核只用于 **FP16 或 W8A16、T=32**（2 或 3 相机），直接修改相应配置中的 `rpu_execution.model.num_cores`。T64–T160 或 Prefill W8A8 使用 paired prefill，chunk 必须为 `(256 × cameras + T) / 2`；默认 `auto` 按输入计算。4/6 核使用通用串行 Vision 路径。

T192（3 相机）用配对 prefill C480（P960 = 2 × 480，2026-11-10 起）：只接受 3 相机、8 核、FP16 / W8A16（没有 C480 的 Prefill W8A8 与 Action NVFP4）。C480 的四块 slab 比 Gemma 的 SPM 预算多 196,608 B，所以入口在 T192 时打开 `RPU_PI05_VISION_LN_STREAM`：SigLIP 27 层的 LayerNorm 权重不再常驻 SPM，每层随输入广播一次，腾出 276,480 B。报告中 `execution_route = "paired_tp8"`，`prefill_chunk_size = 480`。显式写入其他 Vision / Action chunk 或 prefill padding 时不走配对布局，prefill `auto` 为 320 行通用分块（P960 = 3 × 320）。

`[pi05].precision` 另接受 `w4a16_kvint8`：旧的 VLM＋expert per-channel INT4 路径，K/V 投影保留 INT8（不是 KV-cache INT8，也不是 Action NVFP4），权重为 `pi05-libero-finetuned-w4real-kvint8`，只允许 8 核、3 相机、T32。它的公开模板已于 2026-10-04 撤下，不在配置表、支持状态和本轮性能报告中；执行路径和注册权重仍然保留，说明见开发记录。

动作去噪固定 10 步。`input.action_horizon` 取 **50**（默认）或 **10**，它是权重的属性（checkpoint 的 `chunk_size`），配置值与权重不一致时报错；两档使用不同权重，见[权重准备](../WEIGHTS.md)。历史三槽测量使用两路真实相机和一个 `empty_camera_0` 占位槽，输入身份与三路真实相机不同。

<a id="运行"></a>

## 准备与运行

填写 `model.checkpoint` 和 `input.batch`。FP16、W8A16 和 NVFP4 配方各用对应权重，见[权重准备](../WEIGHTS.md)。从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/pi05/2cam/h50/fp16.toml --check-config
python examples/run.py --config examples/configs/pi05/2cam/h50/fp16.toml
```

配置检查核对参数范围；运行时读取 batch 并校验张量形状、mask 和相机名称。

<a id="input-files"></a>

## 输入文件准备

`--check-config` 不读取下列文件，占位路径也能通过；文件不存在或格式不符时在运行时报错。

| 字段 | 内容要求 | 获取方式 |
|---|---|---|
| `input.batch` | `torch.save` 保存的扁平 tensor 字典（顶层直接是下列键）：`observation.images.<相机名>` 每路 `[1,3,256,256]`，相机名集合与 checkpoint 的 `image_features` 一致（2cam 只保留两路真实相机，3cam 含 `empty_camera_0` 占位槽）；`observation.language.tokens` `[1,T]`，T 等于 `input.text_tokens`；bool `observation.language.attention_mask` `[1,T]`，只允许右侧补齐；`observation.state` | 用测试 harness 从 LIBERO 数据集生成，见下方命令 |
| `input.noise`（可选） | 单个 tensor `[1,H,32]`，H 为 action horizon（`h50/` 为 50，`h10/` 为 10），有限实数 | 任意固定噪声即可，例如 H50 用 `torch.save(torch.randn(1,50,32), 'noise.pt')`；不填时按 `run.seed` 生成 |

生成 batch（CPU 即可，不占用 RPU）。需要 `lerobot`、本地 LIBERO 数据集 `lerobot/libero_10_image`、tokenizer `unsloth/gemma-2-2b` 和 Pi0.5 checkpoint。离线环境下，`HF_LEROBOT_HOME` 指向包含 `lerobot/libero_10_image` 的目录，tokenizer 放在 `$RPU_MODEL_CACHE/.hf/hub` 下；数据集找不到时脚本会退回随机数据，此时输出中 `batch_source` 为 `dummy`，不能用于真实输入：

```bash
export RPU_MODEL_CACHE=<含 pi05_libero_finetuned 和 .hf/hub 的目录>
export HF_LEROBOT_HOME=<含 lerobot/libero_10_image 的目录>
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python python/rpu_backend/tests/library/test_pi05_e2e.py --prepare-only \
  --model $RPU_MODEL_CACHE/pi05_libero_finetuned --cameras 3 --tokenizer-max-length 32 \
  --save-batch-cache cache.pt
# 输出中应为 "batch_source": "libero"；取出内层 batch 另存
python -c "import torch; torch.save(torch.load('cache.pt')['batch'], 'pi05_3cam_t32_batch.pt')"
```

`--cameras 2` 生成 2cam batch；`--tokenizer-max-length` 设为所需的 T。2026-10-19 按此流程在板上生成 2cam、3cam 的 T32 batch，并用 `2cam/h50/fp16.toml`、`3cam/h50/fp16.toml`（当时位于相机目录根部）运行完成，输出 `[1,50,7]`。

<a id="kuavo"></a>

## Kuavo 原始观测（openpi pi05）

`kuavo/<构型>/`（真机）和 `kuavo/sim/<构型>/`（仿真）的配置不读 batch，入口按 openpi pi05 的 host 流程从原始观测构造输入（[openpi_inputs.py](../../../examples/runners/pi05/openpi_inputs.py)，依据 LeTools-Learning openpi `eb8687c`）：

- 三路 RGB 帧依次为头部、左腕、右腕（`base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`），以相机原尺寸输入。`[input].frame_size` 写 `[宽, 高]`，即 kuavo_deploy 的 `env.image_size`：真机 `[848, 480]`，仿真 `[640, 480]`；帧尺寸不符时拒绝运行，不会悄悄缩放；
- 每次推理按 openpi `resize_with_pad` 保持宽高比缩放（linear + antialias），补黑边到 224×224：848×480 → 224×126，上下各补 49 行；640×480 → 224×168，上下各补 28 行。模型输入与客户在 4090 上运行的 openpi 流程相同。缩放在 float32 中计算：与 JAX 的融合求和顺序相比，或在不同 CPU（x86 / aarch64）之间，少数像素可能差一个 uint8 级；
- 物理 state 按 norm stats 的 q01/q99 归一化，离散为 256 档整数写入 prompt：`Task: <指令>, State: <整数…>;\nAction: `；
- prompt 用 PaliGemma sentencepiece tokenizer 编码（含 BOS），右侧补齐到 `text_tokens`。超出时拒绝运行，不截断：截断会丢掉 state 或 `Action:` 后缀；
- 输出 `actions.pt` 为归一化动作 `[1,50,action_dim]`。norm stats 含 actions 时另写 `actions_physical.pt`：先反归一化，再把 `delta_action_mask` 为 true 的维度加上当前 state（openpi AbsoluteActions）；
- 计时从解码后的原尺寸帧开始，到物理动作为止：`preprocess`（缩放、离散 state、prompt、tokenize）、Vision / Prefill / Denoise、输出拷回 CPU、`postprocess`（物理动作还原）都在 `total_latency` 内。帧文件解码、加载和预热不计入；相机码流解码见 [H.265 / H.264 硬件解码](#kuavo-video)。

| 构型 | state / action 布局 | action_dim | text_tokens | 模板有效 token | delta_action_mask |
|---|---|---|---|---|---|
| `single_arm` | 右臂 7 + 右手 1 | 8 | 128 | 110 | 7 × true + false |
| `dual_arm` | 左臂 7 + 左手 1 + 右臂 7 + 右手 1 | 16 | 160 | 134 | (7 × true + false) × 2 |
| `wholebody` | `dual_arm` + 下肢 4 | 20 | 192 | 150 | (7 × true + false) × 2 + 4 × true |

`sim/` 下三种构型的布局与上表相同。仿真机器人的末端是 Robotiq 2F-85 夹爪，state 中的“手”维即夹爪开合；仿真模板的有效 token 为 115 / 141 / 157。

短指令模板 `kuavo/single_arm/t64/`、`kuavo/dual_arm/t96/` 及 `sim/` 下的对应目录：布局、权重和 norm stats 与上一级目录相同，指令换成公开数据集 RoboCOIN/Leju_Kuavo_4_box_storage_parcel 的任务原文（`meta/tasks.jsonl`，task_index 0：`moved the parcel from conveyor to table for the left to box.`，含 BOS 14 个 token），`text_tokens` 取任何 state 下都放得下的最小宽度：

| 模板 | 宽度 | 任何 state 下的有效 token | 模板有效 token（真机 / 仿真） | 配对 prefill |
|---|---|---|---|---|
| `single_arm/t64/` | 64 | 39–55 | 49 / 54 | C416 |
| `dual_arm/t96/` | 96 | 55–87 | 73 / 80 | C432 |

全身构型用这条指令需要 63–103 个 token，放不进 T96，所以没有短指令版本。

臂关节单位为 rad；手为 0..1：LeTools 转换数据和 kuavo_deploy 推理时都把灵巧手的 0..100 指令值除以 100（2F-85 夹爪位置除以 0.8）。模板指令含 BOS 共 75 个 token。每个 state 值写成 1–3 位整数，有效 token 数随 state 变化。75 token 指令在 state 最坏情况下需要：单臂 116、双臂 148、全身 165，所以全身模板用 T192（配对 prefill C480，见[输入与核心](#配置矩阵)），单臂、双臂用配对 prefill 的 T128 / T160。

准备输入（CPU 即可）：

```bash
pip install sentencepiece   # 只有原始观测入口需要
curl -L -o paligemma_tokenizer.model https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
# sha256 8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6，openpi 使用的同一文件
huggingface-cli download RoboCOIN/Leju_Kuavo_4_box_storage_parcel --repo-type dataset --local-dir <root> \
  --include "meta/*" "data/*" "videos/chunk-000/*/episode_000000.mp4"
huggingface-cli download LejuRobotics/kuavo_data_challenge_icra sim/TASK2-ParcelWeighing/task2_0001.bag \
  --repo-type dataset --local-dir <sim>                                       # 仿真帧（可选）
python scripts/prepare_pi05_kuavo_inputs.py --dataset <root> --out <dir> \
  --sim-bag <sim>/sim/TASK2-ParcelWeighing/task2_0001.bag   # 需要 pyarrow、av、Pillow；--sim-bag 另需 rosbags
```

公开数据集 [RoboCOIN/Leju_Kuavo_4_box_storage_parcel](https://huggingface.co/datasets/RoboCOIN/Leju_Kuavo_4_box_storage_parcel)（Apache-2.0，LeRobot v2.1）有 311 段、30 fps、三路 848×480 相机。脚本写出 `frames/cam_{h,l,r}.png`（第 0 段第 300 帧，848×480）、`observation.json` 和 `norm_stats/<构型>/norm_stats.json`；给出 `--sim-bag` 时另写 `frames_sim/cam_{h,l,r}.png`（头部相机第 200 帧及时间上最近的两路腕部帧，640×480）。`observation.json` 中各构型的真机和仿真 state 就是模板里的值。norm stats 按 openpi 训练变换统计：10 Hz、50 步 chunk、臂关节为相对当前 state 的 delta、q01/q99。权重和 checkpoint 视图见 [Kuavo 权重准备](../WEIGHTS.md#cmd-pi05-kuavo)。

来源边界：
- 该数据集没有下肢关节，全身构型的 4 个下肢维度使用零 state 和恒等统计（q01=-1、q99=1），只是占位；
- 手部值由数据集的弧度乘 180/π 得到 0..100 指令值，再按 LeTools 除以 100 得到 0..1（2026-11-09 更正：此前误用 0..100，norm stats 的手部维度随之除以 100，模板 prompt 不变）；
- 仿真帧和 state 取自乐聚 ICRA 2026 挑战赛数据 [LejuRobotics/kuavo_data_challenge_icra](https://huggingface.co/datasets/LejuRobotics/kuavo_data_challenge_icra)（`sim/TASK2-ParcelWeighing/task2_0001.bag`，sha256 `70987a5d…a9772c`）：话题与 kuavo_deploy 相同（`/cam_{h,l,r}/color/image_raw/compressed`、`/sensors_data_raw`、`/gripper/state`），数据集卡片未声明许可证。仿真配置沿用真机数据推导的 norm stats；
- 客户的 TrainConfig、norm stats 和微调 checkpoint 不公开，部署时替换 `norm_stats` 和 `[model].checkpoint`；
- 用 pi05_base 和推导的 norm stats 运行只证明运行时正确，不代表任务效果。

<a id="kuavo-server"></a>
## kuavo_server 适配器（在线服务）

`openpi_pi05_rpu` 让 LeTools-Learning 的 `kuavo_server`（ZMQ REP，`kuavo_deploy` 的 `PolicyClient` 连接它）在 RPU 上服务上面的 Kuavo 配置。

组成：
- [`kuavo_server/adapters/openpi_pi05_rpu.py`](../../../examples/runners/pi05/kuavo_server/adapters/openpi_pi05_rpu.py)：放进客户仓库的 drop-in 文件。只导入标准库和本仓库，torch 和 rpu_backend 要等到绑定 CPU 线程和冷选择器之后才加载；
- [`kuavo.py`](../../../examples/runners/pi05/kuavo.py)：观测映射，对齐客户 openpi 适配器 `eb8687c`。内容包括：
  - 头部和腕部相机，缺腕部相机时用头部帧；
  - `which_arm` 的 state / action 切片；
  - `execution_horizon` 截断；
  - `select_action` 的逐步缓存；
- [`serving.py`](../../../examples/runners/pi05/serving.py)：常驻的 `OpenpiPi05Server`。

安装到 LeTools-Learning：
1. 复制 drop-in 文件到 `src/kuavo_server/adapters/openpi_pi05_rpu.py`；
2. 在 `src/kuavo_server/builtin_adapters.py` 的 `ADAPTER_MODULES` 里加一行 `"openpi_pi05_rpu": "kuavo_server.adapters.openpi_pi05_rpu"`；
3. 服务进程要用 RPU 的 Python 环境，以 root 运行（访问 `/dev/rpu`），环境变量与 `examples/run.py` 相同。该环境还需要 `pyzmq`；用 `launch.py` 启动时还需要 `pyyaml`。

启动（全身构型 W8A16）：

```bash
sudo LD_PRELOAD=$(g++ -print-file-name=libstdc++.so.6) <rpu-env>/bin/python src/kuavo_server/serve.py \
  --adapter openpi_pi05_rpu --host 0.0.0.0 --port 5555 \
  --rpu_backend_root <rpu_backend 检出> \
  --rpu_config <rpu_backend>/examples/configs/pi05/kuavo/wholebody/w8a16.toml \
  --prompt "<部署时的任务指令>" --which_arm both
```

用 `launch.py` 时选 `env.mode: none`，再用上面的 root 命令运行 `launch.py openpi_pi05_rpu`，`serve.py` 就继承同一个 Python。`conda run` 不以 root 运行。`launcher.yaml` 片段如下：

```yaml
  openpi_pi05_rpu:
    env:
      mode: none
    args:
      rpu_backend_root: /path/to/rpu_backend
      rpu_config: /path/to/rpu_backend/examples/configs/pi05/kuavo/wholebody/w8a16.toml
      prompt: "flip the express parcel"      # 必须与客户端 task_prompt 一致
      which_arm: both
      nonfinite: hold
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `--rpu_config` | 必填 | Kuavo 示例 TOML：checkpoint、数制、norm stats、tokenizer、相机帧尺寸 `frame_size`、预热用的帧和 state |
| `--rpu_backend_root` | `$RPU_BACKEND_ROOT` | rpu_backend 检出目录 |
| `--prompt` / `--prompts_file` | TOML 的 `instruction` | 启动时准备的指令；文件每行一条 |
| `--which_arm` | `both` | `left` / `right` 时从 16 / 20 维机器人 state 取一侧手臂（加下肢） |
| `--arm_joint_dim` / `--end_effector_dim` | 7 / 1 | 单侧手臂和末端的自由度 |
| `--execution_horizon` | 0 | 每个 chunk 返回的步数；0 表示全部 50 步 |
| `--nonfinite` | `hold` | 输出含 NaN / Inf 时的处理，见下 |
| `--noise` | `per_request` | `per_request`：种子生成器每次请求取新噪声（openpi 行为），`reset` 从头开始；`fixed`：每次都用同一份噪声，等于 `examples/run.py` 的输出 |
| `--seed` | TOML `[run].seed` | 噪声种子 |

指令和有效长度：
- 去噪 Graph 在 BUILD 时固定了 suffix 的 RoPE 起点，即有效前缀长度，所以每个有效 token 数要单独准备一个去噪 Graph；
- 启动时按 tokenizer 推出每条指令在任意 state 下可能的全部长度（每维离散值写成 1–3 位数字），全部预热。真实 tokenizer 下，模板指令的长度范围是：单臂 100–116（17 个），双臂 116–148（33 个），全身 124–164（41 个）；短指令模板为单臂 39–55（17 个）、双臂 55–87（33 个）；
- 冷启动按 `4 + 长度数` 预留 Graph arena（`Pi05Policy.graph_arena_count`）；
- 请求的指令必须是启动时准备过的。`PolicyClient` 默认的 `task_prompt="robot manipulation"` 会被拒绝并返回错误，部署时要传与 `--prompt` 相同的指令。

相机帧尺寸：请求的三路帧必须与 TOML 的 `frame_size` 同尺寸，即 kuavo_deploy 的 `env.image_size`。真机用 `kuavo/<构型>/`，仿真用 `kuavo/sim/<构型>/`；尺寸不符的请求返回错误并计入 `refused`，不会缩放后继续推理。

非有限值处理（NaN / Inf 进入机器人会导致摔倒）：
- state 或浮点相机帧含 NaN / Inf 时，不运行模型，直接返回错误；
- 模型输出（归一化动作或物理动作）含 NaN / Inf 时：`hold` 返回把当前 state 重复 50 步的保持 chunk 并记日志；`error` 返回错误，由客户端抛出 `RuntimeError`。`hold` 要求 action 与 state 同维度；Kuavo 各构型的动作都是 state 布局下的绝对位置目标，满足这一条件；
- `metadata` 返回 `requests` / `nonfinite_outputs` / `refused` 计数。

限制：
- 一个进程只服务一个模型，换配置或指令要重启进程；
- 服务是单线程 ZMQ REP，请求依次执行；
- 没有 `select_action_chunk_rtc`，所以客户端的 RTC 接口不可用。


<a id="kuavo-video"></a>
## 相机 H.265 / H.264 硬件解码（kuavo_deploy）

`kuavo_deploy` 用 `RealtimeH265Decoder` 解码 `/cam_{h,l,r}/color/h265_stream`。原实现在 CPU 上用 PyAV（libavcodec）解码。下面两个文件替换它，改由 R1 的视频解码器（`hx-vpu-dec`）解码：
- [`kuavo_deploy/utils/h265_decoder.py`](../../../examples/runners/pi05/kuavo_deploy/utils/h265_decoder.py)：drop-in。类名、构造参数、`decode` / `is_initialized` / `reset` 和返回值都与原文件相同；
- [`kuavo_deploy/utils/r1_video_decoder.py`](../../../examples/runners/pi05/kuavo_deploy/utils/r1_video_decoder.py)：V4L2 解码器封装。只依赖 numpy 和标准库，也可以单独使用：输出 `i420` 或 `nv12`，`python r1_video_decoder.py --help` 是流检查和测速工具。

安装：把两个文件复制到 LeTools-Learning 的 `src/kuavo_deploy/utils/`，覆盖原 `h265_decoder.py`。运行用户要能读写解码器设备节点，R1 镜像上把用户加入 `video` 组即可；DMA heap `/dev/dma_heap/hx_mem_high_reserve` 对所有用户开放。

RGB 与原实现逐字节相同：
- 硬件只输出 NV12，没有颜色转换（请求 RGB 时仍输出 NV12）；
- RGB 用 PyAV 的同一个 libswscale 调用转换。每个新 SPS 先用 PyAV 解出一帧，记下它的像素格式、色彩矩阵和范围（如 `yuvj420p` / BT.709 全范围），再按这些参数转换硬件输出的 I420；
- 缩放仍是原来的 `cv2.resize(..., INTER_LINEAR)`。

后端选择用环境变量 `KUAVO_VIDEO_DECODER`（或构造参数 `backend=`）：
- `auto`（默认）：有 R1 解码器就用它，否则用 PyAV；
- `r1`：必须用 R1，不可用时报错；
- `pyav`：原实现。

以下情况，该路流自动改用 PyAV（记 warning，`fallback_reason` 写明原因）。出问题的那一帧交给 PyAV 解；已收到的参数集会先送进 PyAV，所以之后的 P 帧要等到下一个 IDR 才有新画面，期间返回上一帧：
- 没有解码器设备；
- 不是 8 位 4:2:0，或是场编码 H.264；
- 超过 `max_size`（默认 1920×1088）；
- 解码连续 3 次超时；
- 画面与 PyAV 的裁剪不一致：FFmpeg 会为对齐把左裁剪向下取整，相机流不会这样裁。

调用 `reset()` 后重新尝试硬件。H.264 流用 `RealtimeH265Decoder(w, h, codec="h264")`。

CPU：客户部署配置的 `image_size` 等于相机尺寸（真机 848×480，仿真 640×480），不缩放。三路真机相机 30 fps 时 R1 路径共 0.25 核，原 PyAV 实现为 0.56 核。`image_size` 与相机尺寸不同时会调用 `cv2.resize`，它默认占用全部 CPU 线程，比解码本身更费 CPU；这时在客户进程启动时调用 `cv2.setNumThreads(1)`，三路总 CPU 从 0.77 核降到 0.42 核（原实现 0.95 核）。该设置作用于整个进程，所以 drop-in 不替客户设置。

延迟：三路共用一个解码器，同时到达的帧依次解码。三路 IDR 同时到达时最后一路约 17 ms，其余帧约 4 ms；每路第一帧要分配帧缓冲，多 50–80 ms。


<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：Pi0.5。

### Pi0.5 相机输入

| 配置 | 支持状态 | 精度 | 量化范围 | cameras 默认值 | action_horizon 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|---|
| [2cam/h10/fp16.toml](2cam/h10/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 2 | 10 | ● 已实测 | ● 已实测 | ● 已实测 | `pi05-libero-h10` |
| [2cam/h10/w8a16.toml](2cam/h10/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action 投影 W8 | 2 | 10 | ● 已实测 | ● 已实测 | ● 已实测 | `pi05-libero-h10-w8a16-vlm-expert` |
| [2cam/h10/w8a16_action_nvfp4.toml](2cam/h10/w8a16_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_action_nvfp4 | Vision / Prefill W8；Action Q/O/Gate/Up/Down NVFP4，Action K/V W8 | 2 | 10 | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_h10_action_nvfp4`；准备方法见[权重准备](../WEIGHTS.md) |
| [2cam/h10/w8a16_prefill_w8a8_action_nvfp4.toml](2cam/h10/w8a16_prefill_w8a8_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_prefill_w8a8_action_nvfp4 | Vision / Prefill W8（17 个非尾层的 Prefill GateUp/GeGLU 为 W8A8）；Action Q/O/Gate/Up/Down NVFP4，Action K/V W8 | 2 | 10 | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_h10_action_nvfp4`；准备方法见[权重准备](../WEIGHTS.md) |
| [2cam/h50/fp16.toml](2cam/h50/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 2 | 50 | ● 已实测 | ● 已实测 | ● 已实测 | `pi05-libero-finetuned` |
| [2cam/h50/w8a16.toml](2cam/h50/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action 投影 W8 | 2 | 50 | ● 已实测 | ● 已实测 | ● 已实测 | `pi05-libero-finetuned-w8a16-vlm-expert` |
| [2cam/h50/w8a16_action_nvfp4.toml](2cam/h50/w8a16_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_action_nvfp4 | Vision / Prefill W8；Action Q/O/Gate/Up/Down NVFP4，Action K/V W8 | 2 | 50 | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_action_nvfp4`；准备方法见[权重准备](../WEIGHTS.md) |
| [2cam/h50/w8a16_prefill_w8a8_action_nvfp4.toml](2cam/h50/w8a16_prefill_w8a8_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_prefill_w8a8_action_nvfp4 | Vision / Prefill W8（17 个非尾层的 Prefill GateUp/GeGLU 为 W8A8）；Action Q/O/Gate/Up/Down NVFP4，Action K/V W8 | 2 | 50 | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_action_nvfp4`；准备方法见[权重准备](../WEIGHTS.md) |
| [3cam/h10/fp16.toml](3cam/h10/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 3 | 10 | ● 已实测 | ● 已实测 | ● 已实测 | `pi05-libero-h10` |
| [3cam/h10/w8a16.toml](3cam/h10/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action 投影 W8 | 3 | 10 | ● 已实测 | ● 已实测 | ● 已实测 | `pi05-libero-h10-w8a16-vlm-expert` |
| [3cam/h10/w8a16_action_nvfp4.toml](3cam/h10/w8a16_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_action_nvfp4 | Vision / Prefill W8；Action Q/O/Gate/Up/Down NVFP4，Action K/V W8 | 3 | 10 | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_h10_action_nvfp4`；准备方法见[权重准备](../WEIGHTS.md) |
| [3cam/h10/w8a16_prefill_w8a8_action_nvfp4.toml](3cam/h10/w8a16_prefill_w8a8_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_prefill_w8a8_action_nvfp4 | Vision / Prefill W8（17 个非尾层的 Prefill GateUp/GeGLU 为 W8A8）；Action Q/O/Gate/Up/Down NVFP4，Action K/V W8 | 3 | 10 | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_h10_action_nvfp4`；准备方法见[权重准备](../WEIGHTS.md) |
| [3cam/h50/fp16.toml](3cam/h50/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 3 | 50 | ● 已实测 | ● 已实测 | ● 已实测 | `pi05-libero-finetuned` |
| [3cam/h50/w8a16.toml](3cam/h50/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action 投影 W8 | 3 | 50 | ● 已实测 | ● 已实测 | ● 已实测 | `pi05-libero-finetuned-w8a16-vlm-expert` |
| [3cam/h50/w8a16_action_nvfp4.toml](3cam/h50/w8a16_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_action_nvfp4 | Vision / Prefill W8；Action Q/O/Gate/Up/Down NVFP4，Action K/V W8 | 3 | 50 | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_action_nvfp4`；准备方法见[权重准备](../WEIGHTS.md) |
| [3cam/h50/w8a16_prefill_w8a8_action_nvfp4.toml](3cam/h50/w8a16_prefill_w8a8_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_prefill_w8a8_action_nvfp4 | Vision / Prefill W8（17 个非尾层的 Prefill GateUp/GeGLU 为 W8A8）；Action Q/O/Gate/Up/Down NVFP4，Action K/V W8 | 3 | 50 | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_action_nvfp4`；准备方法见[权重准备](../WEIGHTS.md) |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 相机 tensor + 文本 tokens + state |
| text_tokens 默认值 | 32 |
| num_steps 默认值 | 10 |
| 文本可调范围 | 32 / 64 / 96 / 128 / 160（存储宽度） |
| 图像/音频可调范围 | 2或3相机；处理后每图[1,3,256,256] |
| 生成可调范围 | num_steps=10；action horizon=10或50（由权重决定） |
| 其他输入形状 | tokens/mask [1,T]；相机名匹配checkpoint |
| 联动限制 | 8核允许2/3相机；4/6核仅 FP16/W8、T32（2或3相机）。T≠32支持 FP16/W8/Action NVFP4（包括 Prefill W8A8 组合）；Action NVFP4 与 Prefill W8A8 要求 chunk=(256×cameras+T)/2（auto 即为此值），FP16/W8 显式写入其他 chunk 时改走通用路径。T192 为配对 prefill C480（P960=2×480）：仅 3 相机、FP16/W8、8 核，SigLIP LayerNorm 改为逐层流式以腾出 SPM；显式写入其他 chunk 时 prefill 用 320 行通用分块（P960=3×320）；板上实测只覆盖 Kuavo 原始观测模板。有效mask计数与物理T分开。 |
| 默认核数 | 8 |
| CPU线程数 | 8（入口默认） |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持（`2cam/h10/w8a16_action_nvfp4.toml`、`2cam/h50/w8a16_action_nvfp4.toml`、`3cam/h10/w8a16_action_nvfp4.toml`、`3cam/h50/w8a16_action_nvfp4.toml`）：原生实现只有 8 核：Action NVFP4 的 striped_v2 权重和张量缩放表只有 8 核布局，少核准入只接受 FP16/W8A16 权重。
- ✕ 4/6 核不支持（`2cam/h10/w8a16_prefill_w8a8_action_nvfp4.toml`、`2cam/h50/w8a16_prefill_w8a8_action_nvfp4.toml`、`3cam/h10/w8a16_prefill_w8a8_action_nvfp4.toml`、`3cam/h50/w8a16_prefill_w8a8_action_nvfp4.toml`）：原生实现只有 8 核：Prefill W8A8 依赖只支持 8 核的配对 prefill 路径，Action 仍为 NVFP4。

输入校验来源：[optimized.py](../../../examples/runners/pi05/optimized.py)。

### Pi0.5 openpi 原始观测（Kuavo）

| 配置 | 支持状态 | 精度 | 量化范围 | text_tokens 默认值 | action_dim 默认值 | frame_size 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|---|---|
| [kuavo/dual_arm/fp16.toml](kuavo/dual_arm/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 160 | 16 | [848,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/dual_arm/t96/fp16.toml](kuavo/dual_arm/t96/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 96 | 16 | [848,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/dual_arm/t96/w8a16.toml](kuavo/dual_arm/t96/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action 投影 W8 | 96 | 16 | [848,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a16-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/dual_arm/w8a16.toml](kuavo/dual_arm/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action 投影 W8 | 160 | 16 | [848,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a16-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/sim/dual_arm/fp16.toml](kuavo/sim/dual_arm/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 160 | 16 | [640,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/sim/dual_arm/t96/fp16.toml](kuavo/sim/dual_arm/t96/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 96 | 16 | [640,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/sim/dual_arm/t96/w8a16.toml](kuavo/sim/dual_arm/t96/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action 投影 W8 | 96 | 16 | [640,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a16-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/sim/dual_arm/w8a16.toml](kuavo/sim/dual_arm/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action 投影 W8 | 160 | 16 | [640,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a16-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/sim/single_arm/fp16.toml](kuavo/sim/single_arm/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 128 | 8 | [640,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a8`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/sim/single_arm/t64/fp16.toml](kuavo/sim/single_arm/t64/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 64 | 8 | [640,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a8`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/sim/single_arm/t64/w8a16.toml](kuavo/sim/single_arm/t64/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action 投影 W8 | 64 | 8 | [640,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a8-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/sim/single_arm/w8a16.toml](kuavo/sim/single_arm/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action 投影 W8 | 128 | 8 | [640,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a8-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/sim/wholebody/fp16.toml](kuavo/sim/wholebody/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 192 | 20 | [640,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a20`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/sim/wholebody/w8a16.toml](kuavo/sim/wholebody/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action 投影 W8 | 192 | 20 | [640,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a20-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/single_arm/fp16.toml](kuavo/single_arm/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 128 | 8 | [848,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a8`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/single_arm/t64/fp16.toml](kuavo/single_arm/t64/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 64 | 8 | [848,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a8`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/single_arm/t64/w8a16.toml](kuavo/single_arm/t64/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action 投影 W8 | 64 | 8 | [848,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a8-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/single_arm/w8a16.toml](kuavo/single_arm/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action 投影 W8 | 128 | 8 | [848,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a8-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/wholebody/fp16.toml](kuavo/wholebody/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 192 | 20 | [848,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a20`；准备方法见[权重准备](../WEIGHTS.md) |
| [kuavo/wholebody/w8a16.toml](kuavo/wholebody/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action 投影 W8 | 192 | 20 | [848,480] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/pi05_base_kuavo_a20-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 3 路 RGB 帧 + 指令 + 物理 state |
| cameras 默认值 | 3 |
| num_steps 默认值 | 10 |
| 文本可调范围 | text_tokens ∈ {32,64,96,128,160,192}（存储宽度；模板 128/160/192，短指令模板 t64/ 与 t96/ 为 64/96，192 仅 3 相机 FP16/W8，配对 prefill C480）；有效 token = BOS + 指令 + 离散 state + “Action:”，超过 T 时拒绝 |
| 图像/音频可调范围 | 3 路 uint8 RGB 相机原尺寸帧（PNG/JPEG/.npy），尺寸须等于 frame_size [宽,高]：真机 [848,480]、仿真 [640,480]；每次推理在计时内按 openpi resize_with_pad 缩放并补黑边到 [1,3,224,224] |
| 生成可调范围 | num_steps=10；action horizon=50；输出 [1,50,action_dim] |
| 其他输入形状 | state 为 1..32 维物理值；openpi norm_stats.json（q01/q99）；PaliGemma sentencepiece tokenizer；delta_action_mask 可选（还原物理动作） |
| 联动限制 | 相机顺序 base / left wrist / right wrist；frame_size 只接受 [848,480] 或 [640,480]，帧尺寸不符时拒绝；checkpoint 的三个 VISUAL 特征须声明 [3,224,224]；action_dim 与 checkpoint output_features.action 一致；不与 batch 同用；[run].reference 只能为 off；T≠32 仅 8 核。 |
| 默认核数 | 8 |
| CPU线程数 | 8（入口默认） |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持：Pi0.5 的 4/6 核只接受 T32：T64–T160 只有 8 核的配对 prefill 布局，T192 的 P960 前缀也只按 8 核绑定。

输入校验来源：[optimized.py](../../../examples/runners/pi05/optimized.py) · [openpi_inputs.py](../../../examples/runners/pi05/openpi_inputs.py)。

<!-- model-configurations-end -->
