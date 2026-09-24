"""CPU-only prefix assembly shared by production and legacy diagnostics."""
import torch


def scatter_visual_embeds_to_dense_cpu(visual_embeds, visual_pos_mask, seq_len, hidden_size):
    """Pure CPU equivalent of qwen3_vl.text.scatter_visual_embeds_to_dense."""
    mask = visual_pos_mask.to(device="cpu", dtype=torch.bool)
    if mask.numel() != seq_len:
        raise ValueError(f"visual mask length {mask.numel()} != seq_len {seq_len}")
    n_vis = int(mask.sum().item())
    dense_list = []
    for i, embed in enumerate(visual_embeds):
        e = embed.detach().to(device="cpu", dtype=torch.float32)
        if e.shape != (n_vis, hidden_size):
            raise ValueError(f"deepstack[{i}] shape {tuple(e.shape)} != {(n_vis, hidden_size)}")
        dense = torch.zeros((seq_len, hidden_size), dtype=torch.float32)
        dense[mask] = e
        dense_list.append(dense.contiguous())
    return dense_list

def prepare_prefix_inputs_from_vision_outputs(model, qin, vision_output):
    """Backend-free assembly: splice pooler + get_rope_index + deepstack-dense.
    `vision_output` may come from CPU or (later) RPU vision; its pooler_output /
    deepstack_features are CPU fp32 with shapes [n_vis, hidden] either way."""
    iface = model.qwen if hasattr(model, "qwen") else model.qwen_vl_interface
    top = iface.model
    outer = top.model
    text_model = outer.language_model
    input_ids = qin["input_ids"]
    attention_mask = qin["attention_mask"]
    image_grid_thw = qin["image_grid_thw"]
    visual_token_id = top.config.image_token_id
    pooler = vision_output.pooler_output
    deepstack_features = list(vision_output.deepstack_features)
    inputs_embeds = text_model.get_input_embeddings()(input_ids).to(torch.float32)
    vmask = (input_ids == visual_token_id)
    vmask_e = vmask.unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(
        vmask_e, pooler.detach().to(device="cpu", dtype=inputs_embeds.dtype)
    )
    mm_token_type_ids = qin.get("mm_token_type_ids", vmask.long())
    qwen_inputs = dict(qin)
    qwen_inputs["mm_token_type_ids"] = mm_token_type_ids
    pos_ids, rope_deltas = outer.get_rope_index(
        input_ids, mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw, video_grid_thw=None, attention_mask=attention_mask,
    )
    hidden_size = inputs_embeds.shape[-1]
    seq_len = input_ids.shape[1]
    vpos = vmask.squeeze(0)
    deepstack_dense = scatter_visual_embeds_to_dense_cpu(deepstack_features, vpos, seq_len, hidden_size)
    prefix_mask = attention_mask.to(torch.bool).cpu()
    return dict(inputs_embeds=inputs_embeds, position_ids=pos_ids, attention_mask=attention_mask,
                deepstack_dense=deepstack_dense, prefix_len=int(attention_mask.sum()),
                prefix_mask=prefix_mask, visual_pos_mask=vmask.cpu(),
                prefix_rope_deltas=rope_deltas.detach().cpu().clone(),
                input_ids=input_ids, qwen_inputs=qwen_inputs, text_model=text_model, top=top, outer=outer)

def get_cache_layer(cache, idx):
    """Extract (key, value) from modern or legacy transformers caches."""
    if hasattr(cache, "layers"):
        layer = cache.layers[idx]
        return layer.keys, layer.values
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[idx], cache.value_cache[idx]
    item = list(cache)[idx]
    return item[0], item[1]
