"""Calibrate and export local activation-aware Qwen3.5 MLP AWQ G128 weights.

The recipe uses actual decoder activations to search equivalent channel scales.
It exports GEMM/direct-zero UINT4, preserves layer zero and non-MLP weights, and
has its own profile identity. It is independent of third-party AWQ checkpoints.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rpu_backend.adapters.qwen3_5.quant_scope import (
    ACTIVATION_AWQ_PROFILE, TEXT_PREFIX, _quanttrio_policy,
    is_full_from_layer_types, quanttrio_awq_config,
)
from .awq_gemm import pack_awq_words
from .precision_recipes import RECORD, write_record

# The one registered precision recipe this converter produces.
FAMILY, RECIPE = 'qwen3_5', 'w4a16_mlp_activation_awq_g128'


def _digest(path):
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_inventory(src):
    """Validate local sharding and bind calibration to the actual source bytes."""
    config = json.loads((src / 'config.json').read_text())
    if any(owner.get(name) for owner in (config, config.get('text_config', {}))
           for name in ('quant_config', 'quantization_config')):
        raise ValueError('AWQ conversion requires unquantized original weights')
    policy = _quanttrio_policy(is_full_from_layer_types(config['text_config']['layer_types']),
                              ACTIVATION_AWQ_PROFILE)
    quanttrio_awq_config(dict(config, quant_config=policy))
    index = src / 'model.safetensors.index.json'
    if index.is_file():
        if (src / 'model.safetensors').exists():
            raise ValueError('ambiguous source checkpoint layout')
        mapping = json.loads(index.read_text())['weight_map']
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('invalid source weight map')
        shards = sorted(set(mapping.values()))
    else:
        mapping, shards = {}, ['model.safetensors']
    if any(not isinstance(p, str) or Path(p).name != p or not p.endswith('.safetensors')
           for p in shards):
        raise ValueError('source shards must be adjacent safetensors files')
    hashes, locations, headers = {}, {}, {}
    for shard in shards:
        path = src / shard
        hashes[shard] = _digest(path)
        with safe_open(path, framework='pt', device='cpu') as tensors:
            for key in tensors.keys():
                if key in locations:
                    raise ValueError('duplicate source tensor')
                locations[key] = shard
                view = tensors.get_slice(key)
                headers[key] = (view.get_dtype(), tuple(view.get_shape()))
    if mapping and mapping != locations:
        raise ValueError('source shard contents disagree with weight map')
    identity = sha256(json.dumps({'config': _digest(src / 'config.json'), 'shards': hashes},
                                sort_keys=True).encode()).hexdigest()
    return config, policy, locations, headers, identity


def quantize_uint4(weight, group_size=128, row_block=256):
    """Asymmetric min/max quantization with representable FP16 scale values."""
    if (weight.dim() != 2 or weight.shape[1] % group_size or row_block < 1
            or not weight.is_floating_point()):
        raise ValueError('AWQ requires floating [N,K] weights with K divisible by G128')
    n, k = weight.shape
    codes = torch.empty((n, k), dtype=torch.uint8)
    scales = torch.empty((k // group_size, n), dtype=torch.float16)
    zeros = torch.empty((k // group_size, n), dtype=torch.uint8)
    for lo in range(0, n, row_block):
        block = weight[lo:lo + row_block].float().reshape(-1, k // group_size, group_size)
        if not bool(torch.isfinite(block).all()):
            raise ValueError('source AWQ weight contains nonfinite values')
        minimum = block.amin(-1).clamp(max=0)
        maximum = block.amax(-1).clamp(min=0)
        # Clamp before casting, then quantize with the stored scale itself.
        scale = ((maximum - minimum) / 15).clamp(
            min=torch.finfo(torch.float16).tiny, max=torch.finfo(torch.float16).max).half()
        zero = torch.round(-minimum / scale.float()).clamp(0, 15)
        q = torch.round(block / scale.float().unsqueeze(-1) + zero.unsqueeze(-1)).clamp(0, 15)
        codes[lo:lo + len(block)] = q.reshape(-1, k).byte().cpu()
        scales[:, lo:lo + len(block)] = scale.t().cpu()
        zeros[:, lo:lo + len(block)] = zero.t().byte().cpu()
    return codes, zeros, scales


def dequantize_uint4(codes, zeros, scales):
    n, k = codes.shape
    group = k // len(scales)
    return ((codes.float().reshape(n, -1, group) - zeros.t().float().unsqueeze(-1))
            * scales.t().float().unsqueeze(-1)).reshape(n, k)


def search_scale(weights, samples, *, grid=9, output_rows=128):
    """Search activation-aware scales against sampled linear output error.

    Output rows are evenly sampled; groups retain every input channel. This
    bounds search memory while the final exporter quantizes all output rows.
    """
    if samples.dim() != 2 or len(samples) < 1 or not bool(torch.isfinite(samples).all()):
        raise ValueError('calibration requires nonempty finite [tokens,channels] activations')
    selected = [weight.detach()[torch.linspace(
        0, weight.shape[0] - 1, min(output_rows, weight.shape[0])).long()].float()
        for weight in weights]
    x = samples.float()
    mean_x = x.abs().mean(0).clamp(min=1e-6)
    mean_w = torch.stack([w.abs().mean(0) for w in selected]).mean(0).clamp(min=1e-6)
    best, best_loss = torch.ones_like(mean_x), float('inf')
    candidates = [torch.ones_like(mean_x)]
    for ratio in torch.linspace(0, 1, grid):
        scale = (mean_x.pow(ratio) / mean_w.pow(1 - ratio)).clamp(1e-3, 1e3)
        candidates.append((scale / torch.sqrt(scale.amin() * scale.amax())).clamp(0.05, 20))
    for scale in candidates:
        loss = 0.0
        for weight in selected:
            q, zero, quant_scale = quantize_uint4(weight * scale)
            error = dequantize_uint4(q, zero, quant_scale) / scale - weight
            loss += float((x @ error.t()).square().mean())
        if loss < best_loss:
            best, best_loss = scale, loss
    return best.contiguous()


def fold_mlp_scales(norm, gate, up, down, input_scale, down_scale):
    """Equivalent full-precision transform, including one-centered RMSNorm."""
    if (input_scale.shape != norm.shape or down_scale.shape != (up.shape[0],)
            or not bool(torch.isfinite(input_scale).all())
            or not bool(torch.isfinite(down_scale).all())
            or not bool((input_scale > 0).all()) or not bool((down_scale > 0).all())):
        raise ValueError('invalid AWQ channel scales')
    return ((1 + norm.float()) / input_scale - 1,
            gate.float() * input_scale,
            up.float() * input_scale / down_scale[:, None],
            down.float() * down_scale)


def _sample_hook(samples, key, limit):
    def hook(_module, args):
        if key not in samples:
            value = args[0].detach().reshape(-1, args[0].shape[-1])
            take = min(limit, len(value))
            rows = torch.linspace(0, len(value) - 1, take, device=value.device).long()
            samples[key] = value[rows].half().cpu().contiguous()
    return hook


@torch.inference_mode()
def calibrate(src, prompts, output, *, device='cpu', max_tokens=128, sample_rows=64):
    """Collect real MLP inputs; no random activations or silently synthetic text."""
    import transformers
    from transformers import AutoTokenizer, AutoModelForImageTextToText

    src, prompts, output = Path(src), Path(prompts), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if max_tokens < 1 or sample_rows < 1:
        raise ValueError("max_tokens and sample_rows must be positive")
    config, _policy, _locations, _headers, identity = source_inventory(src)
    texts = []
    for line in prompts.read_text().splitlines():
        if not line.strip():
            continue
        # JSONL strings or {text: ...}; ordinary text lines are also accepted.
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            value = line
        if isinstance(value, dict):
            value = value.get('text')
        if not isinstance(value, str) or not value.strip():
            raise ValueError('calibration prompts must contain nonempty text')
        texts.append(value)
    if not texts:
        raise ValueError('calibration requires real prompt text')
    tokenizer = AutoTokenizer.from_pretrained(src, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        src, dtype=torch.float16, device_map=device, local_files_only=True).half().eval()
    backbone = model.model.language_model
    samples, handles = {}, []
    # Keep a bounded, deterministic set of rows from each real prompt.
    batches = []
    try:
        for text in texts:
            current = {}
            for i, layer in enumerate(backbone.layers):
                if i == 0:
                    continue
                handles.extend((layer.mlp.register_forward_pre_hook(
                    _sample_hook(current, f'layers.{i}.input', sample_rows)),
                    layer.mlp.down_proj.register_forward_pre_hook(
                    _sample_hook(current, f'layers.{i}.down', sample_rows))))
            tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_tokens)
            ids = tokens['input_ids'].to(next(backbone.parameters()).device)
            batches.append(ids.cpu())
            backbone(input_ids=ids, use_cache=False, return_dict=True)
            for handle in handles:
                handle.remove()
            handles.clear()
            for name, value in current.items():
                samples.setdefault(name, []).append(value)
        expected = {f'layers.{i}.{role}' for i in range(1, config['text_config']['num_hidden_layers'])
                    for role in ('input', 'down')}
        if set(samples) != expected:
            raise ValueError('calibration did not visit every MLP')
        samples = {name: torch.cat(rows) for name, rows in samples.items()}
        metadata = {'format': 'qwen35_awq_calibration_v1', 'source_sha256': identity,
                    'torch_version': str(torch.__version__),
                    'transformers_version': str(transformers.__version__),
                    'prompts_sha256': _digest(prompts),
                    'tokens_sha256': sha256(b''.join(t.numpy().tobytes() for t in batches)).hexdigest()}
        temporary = output.with_name('.' + output.name + f'.tmp-{os.getpid()}')
        try:
            save_file(samples, temporary, metadata=metadata)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        for handle in handles:
            handle.remove()
    return len(texts)


class _ShardWriter:
    def __init__(self, root, limit):
        self.root, self.limit = root, limit
        self.values, self.bytes, self.mapping = {}, 0, {}
        self.index, self.total = 0, 0

    def add(self, name, value):
        value = value.detach().cpu().contiguous()
        size = value.numel() * value.element_size()
        if self.values and self.bytes + size > self.limit:
            self.flush()
        self.values[name] = value
        self.bytes += size

    def flush(self):
        if not self.values:
            return
        self.index += 1
        name = f'model-{self.index:05d}.safetensors'
        save_file(self.values, self.root / name, metadata={'format': 'pt'})
        self.mapping.update({key: name for key in self.values})
        self.total += self.bytes
        self.values, self.bytes = {}, 0

    def finish(self):
        self.flush()
        (self.root / 'model.safetensors.index.json').write_text(json.dumps(
            {'metadata': {'total_size': self.total}, 'weight_map': self.mapping}, indent=2) + '\n')


def convert(src, dst, calibration, *, shard_mb=512, grid=9, row_block=256):
    """Stream source tensors and publish a complete new checkpoint atomically."""
    src, dst, calibration = Path(src).resolve(), Path(dst).resolve(), Path(calibration).resolve()
    if dst.exists():
        raise FileExistsError(dst)
    if shard_mb < 1 or grid < 2 or row_block < 1:
        raise ValueError('shard_mb, grid and row_block must be positive; grid >= 2')
    config, policy, locations, headers, identity = source_inventory(src)
    with safe_open(calibration, framework='pt', device='cpu') as stats:
        metadata = stats.metadata() or {}
        if (metadata.get('format') != 'qwen35_awq_calibration_v1'
                or metadata.get('source_sha256') != identity):
            raise ValueError('calibration is not bound to this original checkpoint')
        calibration_keys = set(stats.keys())
    prefix = TEXT_PREFIX
    layers = config['text_config']['num_hidden_layers']
    expected = {f'layers.{i}.{role}' for i in range(1, layers) for role in ('input', 'down')}
    if calibration_keys != expected:
        raise ValueError('calibration layer inventory differs from source')
    processed = set()
    tmp = dst.with_name('.' + dst.name + f'.tmp-{os.getpid()}')
    if tmp.exists():
        raise FileExistsError(tmp)
    try:
        tmp.mkdir(parents=True)
        writer = _ShardWriter(tmp, shard_mb << 20)
        with ExitStack() as stack:
            files = {name: stack.enter_context(safe_open(src / name, framework='pt', device='cpu'))
                     for name in sorted(set(locations.values()))}
            stats = stack.enter_context(safe_open(calibration, framework='pt', device='cpu'))
            def read(key):
                if key not in locations:
                    raise ValueError(f'missing source tensor {key}')
                value = files[locations[key]].get_tensor(key)
                if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                    raise ValueError(f'nonfloating or invalid source tensor {key}')
                converted = value.half()
                if not bool(torch.isfinite(converted).all()):
                    raise ValueError(f'source tensor overflows FP16: {key}')
                processed.add(key)
                return converted

            for i in range(1, layers):
                base = prefix + f'layers.{i}.'
                names = [base + 'post_attention_layernorm.weight'] + [
                    base + f'mlp.{role}_proj.weight' for role in ('gate', 'up', 'down')]
                norm, gate, up, down = [read(key) for key in names]
                x, z = stats.get_tensor(f'layers.{i}.input'), stats.get_tensor(f'layers.{i}.down')
                if x.shape[1] != gate.shape[1] or z.shape[1] != down.shape[1]:
                    raise ValueError('calibration activation width differs from source MLP')
                input_scale = search_scale((gate, up), x, grid=grid)
                down_scale = search_scale((down,), z, grid=grid)
                norm, gate, up, down = fold_mlp_scales(norm, gate, up, down, input_scale, down_scale)
                norm = norm.half()
                if not bool(torch.isfinite(norm).all()):
                    raise ValueError('AWQ normalization rescaling overflows FP16')
                writer.add(names[0], norm)
                for key, weight in zip(names[1:], (gate, up, down)):
                    if not bool(torch.isfinite(weight.half()).all()):
                        raise ValueError('AWQ rescaling overflows FP16; adjust calibration')
                    codes, zero, scales = quantize_uint4(weight.half(), row_block=row_block)
                    root = key[:-len('.weight')]
                    writer.add(root + '.qweight', pack_awq_words(codes.t().contiguous()))
                    writer.add(root + '.qzeros', pack_awq_words(zero))
                    writer.add(root + '.scales', scales)
                del norm, gate, up, down, x, z
            for key in sorted(locations):
                if key not in processed:
                    writer.add(key, read(key))
        writer.finish()
        config['dtype'] = 'float16'
        config['torch_dtype'] = 'float16'
        for name in ('text_config', 'vision_config'):
            if isinstance(config.get(name), dict):
                config[name]['dtype'] = 'float16'
                config[name]['torch_dtype'] = 'float16'
        config['quant_config'] = policy
        config['quantization_provenance'] = {
            'recipe': ACTIVATION_AWQ_PROFILE, 'source_sha256': identity,
            'calibration_sha256': _digest(calibration),
            'prompts_sha256': metadata['prompts_sha256'],
            'tokens_sha256': metadata['tokens_sha256'],
            'producer': 'convert_qwen3_5_awq', 'scale_search_grid': grid,
            'scope': 'MLP layers 1..last; first layer and attention remain FP16',
            'zero_convention': 'direct', 'rms_norm_weight': 'one_centered',
            'scale_search_objective': 'sampled_linear_output_mse',
            'calibration_runtime': {key: metadata[key] for key in
                                    ('torch_version', 'transformers_version') if key in metadata},
        }
        (tmp / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        for path in src.iterdir():
            if path.is_file() and path.name not in {'config.json', 'model.safetensors.index.json', 'SHA256SUMS', RECORD} \
                    and not path.name.endswith(('.safetensors', '.bin', '.pt', '.pth')):
                shutil.copy2(path, tmp / path.name)
        write_record(tmp, FAMILY, RECIPE)
        tmp.rename(dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {'profile': ACTIVATION_AWQ_PROFILE, 'recipe': RECIPE, 'layers': layers - 1,
            'shards': writer.index}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collect = commands.add_parser('calibrate')
    collect.add_argument('--src', required=True)
    collect.add_argument('--prompts', required=True)
    collect.add_argument('--output', required=True)
    collect.add_argument('--device', default='cpu')
    collect.add_argument('--max-tokens', type=int, default=128)
    collect.add_argument('--sample-rows', type=int, default=64)
    export = commands.add_parser('convert')
    export.add_argument('--src', required=True)
    export.add_argument('--dst', required=True)
    export.add_argument('--calibration', required=True)
    export.add_argument('--shard-mb', type=int, default=512)
    export.add_argument('--grid', type=int, default=9)
    export.add_argument('--row-block', type=int, default=256)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    result = calibrate(**args) if command == 'calibrate' else convert(**args)
    print(json.dumps({'status': 'complete', 'result': result}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
