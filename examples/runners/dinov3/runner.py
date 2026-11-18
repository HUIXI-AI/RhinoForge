"""Public dinov3 model construction."""
from __future__ import annotations

from examples.common.text import text_context

def prepare(config, runner):
    import torch
    from examples.common.generation import _image_inputs
    example, inputs, target, checkpoint, execution, decode_steps = text_context(config)
    from transformers import AutoConfig, AutoImageProcessor, DINOv3ViTModel
    from rpu_backend.adapters.dinov3 import DINOv3Adapter
    from examples.common.config import install_quantization

    images = _image_inputs(inputs)
    hf_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    DINOv3Adapter.preflight(hf_config)
    quantization = install_quantization(target, example["profile_id"])
    if quantization is not None:
        # Refuse the recipe before reading the checkpoint.
        DINOv3Adapter.preflight_quantization(hf_config, execution, quantization)
    processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    pixels = processor(images=images, return_tensors="pt").pixel_values
    runner.input_geometry = {"source_image_sizes_wh": [list(image.size) for image in images],
                             "pixel_values_shape": list(pixels.shape), "cameras": len(images)}
    model = DINOv3ViTModel.from_pretrained(
        checkpoint, config=hf_config, dtype=torch.float16,
        local_files_only=True).eval()
    model = DINOv3Adapter(model, rpu_execution=execution, quantization=quantization).to_rpu()
    runner.owner = model

    def infer_vision():
        output = model(pixels, _rpu_cpu_post_layer_norm=True)
        hidden_cpu = output.last_hidden_state.cpu()
        return {"last_hidden_state": hidden_cpu,
                "pooler_output": hidden_cpu[:, 0, :].clone()}

    return infer_vision, model
