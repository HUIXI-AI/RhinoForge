"""Hy-Embodied-0.5-VLA 的 host 预处理 —— 原始观测 → `HyVlaRunner.get_action` 的入参。

移植自 vendor `hy_vla/modeling_hy_vla.py` 的
`prepare_images` / `prepare_language` / `prepare_state` / `make_att_2d_masks` /
`embed_prefix` / `embed_suffix` / `_apply_visual_segment_mask` /
`sample_actions` / `denoise_step`。

本模块只搬 **mask、position_ids 与张量整形**；embedding 查表仍由
`HyVlaRunner.assemble_prefix` 做。vendor 的 `embed_prefix` 把"排 token 布局"和
"查 embedding"写在同一个函数里，这里只要前者 —— 布局（谁在第几行、哪几行是
vision）恰恰就是 mask 的定义域。

⚠️ 常量全部来自交付 ckpt 的 `config.json`，**不要手改**：

    visual_segment_isolation = False    # patch-only 分支
    max_state_dim            = 32
    max_action_dim           = 32
    n_action_steps           = 50
    tokenizer_max_length     = 64
    resize_imgs_with_padding = [224, 224]
    empty_cameras            = 0        # ⇒ 缺相机不会补空图，直接少一段
    image_features 的键序     = top_head, hand_left, hand_right   ← 相机顺序

⚠️ 产出的 prefix 侧张量恒为 **240 行**（2 + 3×58 + 64），与 `prefix_len=S` 无关 ——
`HyVlaRunner` 自己按 `[:S]` 切（`runtime.py` 的 `prefill_mask[:S,:S]` /
`cat([denoise_mask[:,:S], denoise_mask[:,-51:]])`）。所以这里不要提前切。
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

# config.json 的 `image_features` 键序 —— 就是 `prepare_images` 遍历的顺序，
# 也就是调用方必须按这个顺序给图。换序不会报错，只会静默算错。
CAMERA_ORDER = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)

# vendor `embed_prefix` 的固定前缀：<hy_begin_of_sentence><hy_User>
_N_LEAD = 2


def resize_with_pad(img: torch.Tensor, width: int, height: int,
                    pad_value: float = 0.0, mode: str = "bilinear") -> torch.Tensor:
    """等比缩放到框内 + 居中补边。逐字移植 vendor 的同名函数。

    ⚠️ `int()` 截断（不是 round）与"补在中心、余数补在下/右"这两点都照抄 ——
    480×640 → ratio=max(640/224,480/224)=2.857 ⇒ 168×224，上下各补 28 行。
    改成 round 或补在一侧，patch 网格就整体错位。
    """
    if img.ndim != 4:
        raise ValueError(f"期望 (b,c,h,w)，得到 {tuple(img.shape)}")
    cur_h, cur_w = img.shape[2:]
    ratio = max(cur_w / width, cur_h / height)
    resized_h, resized_w = int(cur_h / ratio), int(cur_w / ratio)

    params = {"size": (resized_h, resized_w), "mode": mode}
    if mode != "nearest":
        params["align_corners"] = False
    out = F.interpolate(img, **params)

    pad_h = max(0, int(height - resized_h))
    pad_w = max(0, int(width - resized_w))
    pw, ph = pad_w // 2, pad_h // 2
    return F.pad(out, (pw, pad_w - pw, ph, pad_h - ph), value=pad_value)


def _as_float01_nchw(frame, index: int) -> torch.Tensor:
    """一帧任意常见形态的相机图 → `[1,3,H,W]` fp32、像素在 **[0,1]**。

    收下这些（都是真机客户端实际会送的形态）：

    | 入参 | 处理 |
    |---|---|
    | `[1,3,H,W]` / `[3,H,W]` fp32 ∈ [0,1] | 原样（**逐位不变**，老路径就是它）|
    | `[H,W,3]` / `[1,H,W,3]` | `permute` 成 CHW |
    | `uint8` / 其他整型（0..255） | `/255` |
    | `numpy.ndarray` | `torch.from_numpy` |
    | PIL Image（任何带 `.convert` 的对象）| `convert("RGB")` → HWC uint8 |
    | 浮点但值域越界 | **抛 `ValueError`** |

    ⚠️ 这层是**门禁不是转换器**：唯一会被"猜"的是 CHW/HWC 布局，其余都是无歧义的。
    浮点帧越界（例如 0..255 的 float）**不自动 /255** —— 分不清是"忘了归一"还是
    "已经是 [-1,1] 的模型输入"，猜错就是静默算错。整型的 0..255 才是无歧义的，
    才自动除。这条是本函数存在的理由：改之前 CHW uint8 会**一路算到底**，
    产出值域 `[-1.0, 503.2]` 的"图"，不报错、只是全错。

    ⚠️ 布局二义只可能出现在 3×3 像素这种病态尺寸上，规则是 **CHW 优先**
    （`shape[1] == 3` 先判）。
    ⚠️ **通道序必须是 RGB**，这里不做任何色彩空间修正 —— 送 BGR 不报错、只是
    静默算错（与 `serve.py` / lingbot2 的 numpy 输入约定一致，通道序归调用方）。
    """
    import numpy as np

    src = frame
    if not torch.is_tensor(src) and not isinstance(src, np.ndarray):
        convert = getattr(src, "convert", None)          # PIL.Image 或其鸭子类型
        if convert is None:
            raise TypeError(f"第 {index} 台相机：不认识的帧类型 {type(src).__name__}，"
                            f"需要 torch.Tensor / numpy.ndarray / PIL.Image")
        src = np.array(convert("RGB"))                   # 可写副本；asarray 会给只读 buffer
    if isinstance(src, np.ndarray):
        src = torch.from_numpy(np.ascontiguousarray(src))

    if src.ndim == 3:
        img = src[None]
    elif src.ndim == 4:
        if src.shape[0] != 1:
            raise ValueError(f"第 {index} 台相机：一次只收一帧，得到 batch={src.shape[0]}")
        img = src
    else:
        raise ValueError(f"第 {index} 台相机：需要 3 维或 4 维，得到 {tuple(src.shape)}")

    if img.shape[1] == 3:
        pass                                             # 已是 NCHW
    elif img.shape[-1] == 3:
        img = img.permute(0, 3, 1, 2).contiguous()       # NHWC → NCHW
    else:
        raise ValueError(f"第 {index} 台相机：需要 3 通道 RGB，得到 {tuple(src.shape)}")

    if img.dtype == torch.bool:
        raise ValueError(f"第 {index} 台相机：bool 帧无法解释为像素")
    if not img.is_floating_point():
        return img.float().div_(255.0)                   # .float() 已换 dtype ⇒ 是新张量
    img = img.float()
    lo, hi = torch.aminmax(img)
    if float(hi) > 1.5 or float(lo) < -0.01:
        raise ValueError(
            f"第 {index} 台相机：浮点帧的像素范围是 [{float(lo):.4g}, {float(hi):.4g}]，"
            f"不在 [0,1]。本层只收归一化后的浮点；0..255 的浮点请自行 `/255`"
            f"（整型的 0..255 会自动除，浮点的不猜）")
    return img


def prepare_images(frames: Sequence, size: int = 224) -> list[torch.Tensor]:
    """每相机一帧 → `[1,3,224,224]` ∈ [-1,1]。入参形态见 :func:`_as_float01_nchw`。

    两步都照 vendor：`resize_with_pad(..., pad_value=0)` 然后 `img*2-1`
    （SigLIP 系的输入约定）。补边区因此落在 **-1**，不是 0。

    对 `[1,3,H,W]`、范围为 [0,1] 的 fp32 输入，范围检查不改变数值，
    `.float()` 也不进行 dtype 转换。
    """
    out = []
    for i, frame in enumerate(frames):
        img = _as_float01_nchw(frame, i)
        img = resize_with_pad(img, size, size, pad_value=0.0)
        out.append(img * 2.0 - 1.0)
    return out


def prepare_language(instruction: str, tokenizer, max_length: int = 64
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """指令字符串 → lang_tokens [1,L] int64 与 lang_masks [1,L] bool。

    清洗指令中的下划线和换行，补齐 <｜hy_Assistant｜> 后缀，再按右侧
    定长 padding 分词。add_special_tokens=False：prefix 会自行添加 BOS。
    先移除已有后缀，再清洗指令本身，最后补回后缀；否则后缀中的 ASCII
    下划线会被替换成空格，导致重复后缀和额外 token。"""
    tag = "<｜hy_Assistant｜>"
    task = instruction.strip()
    if task.endswith(tag):
        task = task[: -len(tag)].strip()
    task = task.replace("_", " ").replace("\n", " ")
    task = f"{task}{tag}"
    tok = tokenizer(
        [task],
        padding="max_length",
        padding_side="right",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return tok["input_ids"].long(), tok["attention_mask"].bool()


def prepare_state(state: torch.Tensor | Sequence[float] | None,
                  max_state_dim: int = 32) -> torch.Tensor:
    """本体感受 → [1, max_state_dim] FP32，右侧补零。

    None 表示全零 state。全零输入经过 state_proj 后只保留 bias；
    非零输入仍由完整权重矩阵参与投影。"""
    out = torch.zeros(1, max_state_dim, dtype=torch.float32)
    if state is None:
        return out
    s = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
    if s.shape[1] > max_state_dim:
        raise ValueError(f"state 有 {s.shape[1]} 维，超过 max_state_dim={max_state_dim}")
    out[:, : s.shape[1]] = s
    return out


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """vendor 同名函数（源自 big_vision）：`att_masks` 的 cumsum 定义分块因果。

    `cumsum[j] <= cumsum[i]` ⇒ i 能看见 j。prefix 侧 `att_masks` 全 1 ⇒ 退化成
    纯因果；suffix 侧 `[1,1,0,…,0]` ⇒ 50 个 action token 同属一块、彼此全可见。
    """
    if att_masks.ndim != 2 or pad_masks.ndim != 2:
        raise ValueError("pad_masks / att_masks 都必须是 2 维")
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


def prefix_masks(n_images: int, grid: int, lang_masks: torch.Tensor):
    """prefix 的三条 1-D mask + 图像 token 的行区间。

    布局（vendor `embed_prefix`）：

        <bos><hy_User>
        每图： <vision_start> + grid 行 ×(grid 个 patch + 1 个 split) + <vision_end>
        语言 token（含 padding）

    grid=7 ⇒ 每图 1+56+1 = 58 行；3 图 + 2 前缀 + 64 语言 = **240**。

    Returns:
        pad_masks / att_masks / modality_mask：均 `[1, N]`；
        image_idx_ranges：每行 patch 的 `[start, end)`（**不含**行尾 split）；
        image_full_ranges：每图 patch+split 的整段 `[start, end)`（不含 start/end 标记）。
    """
    row_len = grid + 1                 # grid 个 patch + 1 个 split
    per_img = grid * row_len           # 56
    n_lang = int(lang_masks.shape[1])

    att: list[int] = [1, 1]            # bos, user
    modality: list[bool] = [False, False]
    pad_parts = [torch.ones(1, _N_LEAD, dtype=torch.bool)]
    idx_ranges: list[tuple[int, int]] = []
    full_ranges: list[tuple[int, int]] = []

    for _ in range(n_images):
        att.append(1)                  # vision_start
        modality.append(False)
        pad_parts.append(torch.ones(1, 1, dtype=torch.bool))

        start = len(att)               # ⚠️ vendor 在 append(vision_start) **之后**取
        idx_ranges.extend((start + r * row_len, start + r * row_len + grid)
                          for r in range(grid))
        full_ranges.append((start, start + per_img))

        att.extend([1] * per_img)
        # 每行：grid 个 patch(modality=True) + 1 个 split(False)
        modality.extend(([True] * grid + [False]) * grid)
        pad_parts.append(torch.ones(1, per_img, dtype=torch.bool))

        att.append(1)                  # vision_end
        modality.append(False)
        pad_parts.append(torch.ones(1, 1, dtype=torch.bool))

    att.extend([1] * n_lang)
    modality.extend([False] * n_lang)
    pad_parts.append(lang_masks.bool())   # ← 只有这一段可能是 False（prompt padding）

    pad_masks = torch.cat(pad_parts, dim=1)
    att_masks = torch.tensor(att, dtype=torch.bool)[None]
    modality_mask = torch.tensor(modality, dtype=torch.bool)[None]
    return pad_masks, att_masks, modality_mask, idx_ranges, full_ranges


def apply_visual_segment_mask(att_2d: torch.Tensor,
                              image_idx_ranges: Sequence[tuple[int, int]],
                              image_full_ranges: Sequence[tuple[int, int]],
                              isolation: bool = False) -> None:
    """**原地**改写 2-D mask 的视觉段（vendor `_apply_visual_segment_mask`）。

    只实现 `isolation=False`（patch-only）—— 交付 ckpt 的
    `config.json: visual_segment_isolation=False`。两步：
      ① 把**所有**图的 patch token 之间的可见性清零（连因果通路一起断）；
      ② 再把**同一张图内**的 patch token 之间打开成双向。
    ⇒ 跨图的 patch 互不可见，图内 patch 全可见；split / vision_start / vision_end
    仍走因果通路。

    `isolation=True`（RoboTwin post-train ckpt 用的全段隔离）**故意不实现**：
    它会改变数值，而本仓没有对应的 golden 可验。遇到就显式报错，别静默算错。
    """
    if isolation:
        raise NotImplementedError(
            "visual_segment_isolation=True（全段隔离）未移植：它与本交付 ckpt 的 "
            "config.json(False) 不同，且本仓没有该模式的 golden。参见 vendor "
            "modeling_hy_vla.py::_apply_visual_segment_mask 的 True 分支。")

    all_idx: list[int] = []
    for s, e in image_idx_ranges:
        all_idx.extend(range(s, e))
    if all_idx:
        idx = torch.tensor(all_idx)
        att_2d[:, idx[:, None], idx[None, :]] = False

    for f_start, f_end in image_full_ranges:
        own = [i for s, e in image_idx_ranges if s >= f_start and e <= f_end
               for i in range(s, e)]
        if own:
            idx = torch.tensor(own)
            att_2d[:, idx[:, None], idx[None, :]] = True


def suffix_masks(n_action: int = 50, n_state_tokens: int = 1):
    """suffix（1 个 state token + 50 个 action token）的三条 1-D mask。

    `att_masks = [1] + [0]*(T_state-1)` 再接 `[1] + [0]*(n_action-1)` ——
    state 自成一块、action chunk 自成一块，块内双向、块间因果（vendor
    `embed_suffix`）。`modality_mask` 全 True：两段都走 `_v` 塔。
    """
    total = n_state_tokens + n_action
    att = [1] + [0] * (n_state_tokens - 1) + [1] + [0] * (n_action - 1)
    return (torch.ones(1, total, dtype=torch.bool),
            torch.tensor(att, dtype=torch.bool)[None],
            torch.ones(1, total, dtype=torch.bool))


def sample_noise(n_action: int = 50, action_dim: int = 32,
                 seed: int | None = None) -> torch.Tensor:
    """flow-matching 的初始 `x_t`，`[1, n_action, action_dim]` fp32。

    `seed=None` ⇒ 用全局 RNG（对齐 vendor 的 `torch.normal`）；给 seed 则用独立
    `Generator`，**不动全局 RNG 状态**，同 seed 逐位可复现。noise 是输入的一部分，
    对 golden 比精度时必须与参考实现同源。
    """
    shape = (1, n_action, action_dim)
    if seed is None:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32)
    g = torch.Generator().manual_seed(int(seed))
    return torch.normal(mean=0.0, std=1.0, size=shape, generator=g, dtype=torch.float32)


def build_observation(frames: Sequence[torch.Tensor], instruction: str, tokenizer,
                      *, state=None, noise: torch.Tensor | None = None,
                      noise_seed: int | None = None, grid: int = 7,
                      max_state_dim: int = 32, max_action_dim: int = 32,
                      n_action: int = 50, tokenizer_max_length: int = 64,
                      image_size: int = 224) -> dict:
    """原始观测 → `HyVlaRunner.get_action(**obs)` 的全部入参 + `n_valid`。

    返回的 dict 除 `get_action` 的 9 个键外还带一个 **`n_valid`** —— prefix 的有效
    行数，调用方据此断言 `ceil16(n_valid) <= prefix_len`（超了就是 prompt 太长，
    应当在建 runner 时就用更大的 S，而不是让它静默截断）。
    """
    images = prepare_images(frames, size=image_size)
    lang_tokens, lang_masks = prepare_language(instruction, tokenizer,
                                               max_length=tokenizer_max_length)

    pad_p, att_p, mod_p, idx_ranges, full_ranges = prefix_masks(
        len(images), grid, lang_masks)
    prefill_mask = make_att_2d_masks(pad_p, att_p)
    apply_visual_segment_mask(prefill_mask, idx_ranges, full_ranges)
    prefill_pos = torch.cumsum(pad_p, dim=1) - 1

    pad_s, att_s, mod_s = suffix_masks(n_action)
    suffix_2d = make_att_2d_masks(pad_s, att_s)
    # denoise 的 prefix 块是 pad_masks 的广播（不是因果）—— suffix 每一行都能看见
    # 全部有效 prefix；padding 列靠 pad_masks 关掉。
    prefix_2d = pad_p[:, None, :].expand(1, pad_s.shape[1], pad_p.shape[1])
    denoise_mask = torch.cat([prefix_2d, suffix_2d], dim=2)
    # suffix 的 position_ids 从 prefix 的**有效**长度接着数（不是从 S）。
    suffix_pos = torch.sum(pad_p, dim=-1)[:, None] + torch.cumsum(pad_s, dim=1) - 1

    if noise is None:
        noise = sample_noise(n_action, max_action_dim, seed=noise_seed)

    return {
        "images": images,
        "lang_tokens": lang_tokens[0],
        "prefill_mask": prefill_mask[0],
        "prefill_pos": prefill_pos[0],
        "modality_mask": mod_p[0],
        "denoise_mask": denoise_mask[0],
        "suffix_pos": suffix_pos[0],
        "noise": noise.float(),
        "state": prepare_state(state, max_state_dim),
        "n_valid": int(pad_p.sum()),
    }
