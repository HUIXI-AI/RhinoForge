"""LingBot-VLA (robbyant/lingbot-vla-4b) 的独立 RPU adapter。

组合 Qwen2.5-VL VLM 与基于 Qwen2 decoder 的 AdaRMS flow-matching expert。
文本 decoder 和 expert 在 RPU 执行。该 checkpoint 的 FP16 vision
存在溢出边界，因此本 adapter 保留 CPU FP32 vision 路径。"""
from rpu_backend.adapters.lingbot_vla.runtime import LingbotVlaPolicy, build_lingbot_vla, run_vit

__all__ = ["LingbotVlaPolicy", "build_lingbot_vla", "run_vit"]
