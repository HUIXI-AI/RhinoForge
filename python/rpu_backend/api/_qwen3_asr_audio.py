"""Bounded ASR audio blocks on the shared fused bidirectional encoder."""
import torch
import torch.nn.functional as F

from .cache import RPUCache
from .errors import UnsupportedModelError
from rpu_backend.runtime.weights import transform_linear_weight


def _pad_attention(weight, bias, *, output=False):
    # Physical attention has sixteen D64 heads; keep the original residual width.
    missing = 1024 - weight.shape[1 if output else 0]
    if missing not in (0, 128):
        raise UnsupportedModelError("ASR attention requires fourteen or sixteen heads")
    if output:
        return F.pad(weight, (0, missing)), bias
    return F.pad(weight, (0, 0, 0, missing)), F.pad(bias, (0, missing))


class ResidentAudioEncoder:
    def __init__(self, audio, counters, *, num_cores=8):
        if type(num_cores) is not int or num_cores not in (4, 8):
            raise ValueError("ASR audio encoder requires four or eight cores")
        self.num_cores = num_cores
        self.audio = audio
        self.counters = counters
        self.handle = None
        self.graph = None
        self.cache = None
        self.last_length = None
        self.last_execution_plan = None
        self.last_input_geometry = None
        self.calls = 0
        self.positions = None
        self._resource = None

    def install(self, parent=None):
        from rpu_backend.runtime._native_retirement import _InstalledNativeResource
        from rpu_backend.graph import GraphRuntimePolicy

        a = self.audio
        c = a.config
        if (c.d_model, c.encoder_layers, c.encoder_attention_heads, c.encoder_ffn_dim,
                c.activation_function, c.n_window, c.n_window_infer, c.output_dim) not in (
                (896, 18, 14, 3584, "gelu", 50, 800, 1024),
                (1024, 24, 16, 4096, "gelu", 50, 800, 2048)):
            raise UnsupportedModelError("resident ASR audio geometry differs")
        if len(a.layers) != c.encoder_layers:
            raise UnsupportedModelError("resident ASR audio layer count differs")
        for layer in a.layers:
            if layer.self_attn_layer_norm.eps != 1e-5 or layer.final_layer_norm.eps != 1e-5:
                raise UnsupportedModelError("resident ASR LayerNorm epsilon differs")
        if a.ln_post.eps != 1e-5:
            raise UnsupportedModelError("resident ASR final LayerNorm epsilon differs")
        self.graph = torch.rpu.GraphCache(max_entries=1, runtime_policy=
            GraphRuntimePolicy.from_environment(execution_core_count=self.num_cores))
        self.handle = torch.ops.rpu.siglip_create(True)
        columns = [[] for _ in range(16)]
        keepalive = {"audio": a, "weights": columns}
        self._resource = _InstalledNativeResource(
            self, self.handle, torch.ops.rpu.siglip_destroy,
            graphs=(self.graph,), keepalive=keepalive,
            label="ASR audio encoder", handle_name="handle",
        )
        self._resource.take_ownership(self if parent is None else parent)
        if self.num_cores != 8:
            torch.ops.rpu.siglip_set_execution_core_count(self.handle, self.num_cores)
        for layer in a.layers:
            attn = layer.self_attn
            modules = [attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj, layer.fc1, layer.fc2]
            biases = []
            for i, module in enumerate(modules):
                weight, bias = module.weight.detach(), module.bias.detach()
                if i < 4:
                    weight, bias = _pad_attention(weight, bias, output=i == 3)
                partition = 0 if i in (3, 5) else 1
                columns[i].append(transform_linear_weight(weight.contiguous(), partition, self.num_cores).to("rpu"))
                biases.append(bias.contiguous().to("rpu"))
            for i, tensor in enumerate([layer.self_attn_layer_norm.weight, layer.self_attn_layer_norm.bias,
                                       layer.final_layer_norm.weight, layer.final_layer_norm.bias]):
                columns[6 + i].append(tensor.detach().to("rpu"))
            for i, bias in enumerate(biases):
                columns[10 + i].append(bias)
        torch.ops.rpu.siglip_set_weights(self.handle, *columns,
            a.ln_post.weight.detach().to("rpu"), a.ln_post.bias.detach().to("rpu"),
            transform_linear_weight(a.proj1.weight.detach(), 1, self.num_cores).to("rpu"),
            a.proj1.bias.detach().to("rpu"), 16, 64, c.d_model, c.encoder_ffn_dim, c.d_model, 1e-5)
        torch.ops.rpu.siglip_set_audio_mode(self.handle)
        torch.ops.rpu.siglip_set_audio_projection(self.handle,
            transform_linear_weight(a.proj2.weight.detach(), 1, self.num_cores).to("rpu"),
            a.proj2.bias.detach().to("rpu"))
        actual = tuple(torch.ops.rpu.siglip_get_execution_topology(self.handle))
        expected = (1, self.num_cores, self.num_cores, self.num_cores, 1, 8,
                    c.encoder_ffn_dim, c.encoder_ffn_dim)
        if actual != expected:
            raise RuntimeError("ASR audio native topology differs from its installed weights")
        self.cache = RPUCache(c.encoder_layers, 1, 128, 16, 64, dtype=torch.float16, attn_tp=self.num_cores)
        self.positions = a.positional_embedding.positional_embedding[:13].unsqueeze(0).expand(8, -1, -1).contiguous().to("rpu")
        keepalive.update(cache=self.cache, positions=self.positions)

    def encode(self, hidden):
        from rpu_backend.adapters.siglip import _siglip_component_execution_plan

        length = hidden.shape[0]
        width = self.audio.config.d_model
        if hidden.shape != (length, width) or not 7 <= length <= 104:
            raise ValueError(f"resident ASR requires 7–104 audio tokens of width {width}")
        if self.last_length != length:
            self.graph.clear()
        self.last_length = length
        plan = _siglip_component_execution_plan(self.audio, self.handle, length, 1,
            external_patch_prologue=False, graph_cache=self.graph)
        descriptor = list(plan.selected.stage_tuple.physical_descriptor)
        signature = torch.rpu.GraphSignature(op_id="qwen3_asr_audio", shapes=[length, width],
            dyn_dims=[self.handle, *plan.graph_key_words()], dtypes=[torch.float16])
        inputs = hidden.unsqueeze(0).contiguous().to("rpu")
        try:
            with self.graph.capture(signature):
                result = torch.ops.rpu.siglip_forward(self.handle, inputs,
                    list(self.cache.k_caches), list(self.cache.v_caches), descriptor)
            result = result[0]
        finally:
            torch.ops.rpu.spm_alloc_reset_temporary()
        for name in self.counters:
            if name.startswith("layers.") or name in ("proj1", "proj2"):
                self.counters[name] += 1
        self.calls += 1
        # Keep the successful selected plan; execution_stats serializes it after timing.
        self.last_execution_plan = plan
        return result

    def forward(self, input_features, feature_lens=None, aftercnn_lens=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        a = self.audio
        if (input_features.device.type != "cpu" or input_features.dtype != torch.float16 or
                input_features.ndim != 2 or input_features.shape[0] != 128 or
                not 50 <= input_features.shape[1] <= 800 or feature_lens is None or
                feature_lens.numel() != 1 or int(feature_lens.item()) != input_features.shape[1]):
            raise ValueError("resident ASR requires one CPU FP16 mel input with 50–800 frames")
        # shortcut: one <=8-second utterance fits a single 800-frame attention window.
        chunks = list(input_features.T.split(100))
        self.last_input_geometry = {"mel_shape": list(input_features.shape),
            "conv_chunk_frames": [int(chunk.shape[0]) for chunk in chunks],
            "conv_padded_frames": max(int(chunk.shape[0]) for chunk in chunks),
            "torch_num_threads": torch.get_num_threads(),
            "inference_mode": torch.is_inference_mode_enabled()}
        padded = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2).unsqueeze(1)
        hidden = F.gelu(a.conv2d1(padded))
        hidden = F.gelu(a.conv2d2(hidden))
        hidden = F.gelu(a.conv2d3(hidden))
        b, c, f, t = hidden.shape
        hidden = a.conv_out(hidden.permute(0, 3, 1, 2).contiguous().view(b, t, c * f).to("rpu"))
        hidden = hidden + self.positions[:b, :t]
        # Only the final 100-frame chunk has padding, so valid rows form one prefix.
        tokens = sum((chunk.shape[0] + 7) // 8 for chunk in chunks)
        hidden = hidden.reshape(-1, a.config.d_model)[:tokens]
        hidden = self.encode(hidden)
        return BaseModelOutputWithPooling(last_hidden_state=hidden)

    def close(self):
        if self._resource is not None:
            self._resource.retire()
        elif self.graph is not None:
            self.graph.clear()
        self.cache = self.graph = None
        self.positions = None
