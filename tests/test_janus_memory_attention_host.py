"""Board-free tests; import leaves the device runtime unloaded."""
import importlib.util
from pathlib import Path
import unittest
import sys
import types
import threading
import json
import struct
import tempfile
import hashlib
import io
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1] / "python/rpu_backend/adapters/janus_vln"


def load(name):
    package = types.ModuleType("_janus_test")
    package.__path__ = [str(ROOT)]
    sys.modules.setdefault("_janus_test", package)
    spec = importlib.util.spec_from_file_location("_janus_test." + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class JanusTests(unittest.TestCase):
    def test_streamed_tensor_integrity_and_truncation(self):
        module = load("checkpoint")
        payload = torch.arange(4).half().numpy().tobytes()
        manifest = {"base_url": "http://127.0.0.1:18765", "shards": {
            "part.safetensors": {"header_length": 100, "header": {
                "weight": {"shape": [4], "dtype": "F16", "data_offsets": [0, 8]}}}},
            "tensor_sha256": {"weight": hashlib.sha256(payload).hexdigest()}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {"weight": "part.safetensors"}}))
            (root / "config.json").write_text("{}")
            manifest.update(checkpoint_revision="8631eee38a4c688256473ec6a59917fb5eec58ff",
                config_sha256=hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
                index_sha256=hashlib.sha256((root / "model.safetensors.index.json").read_bytes()).hexdigest())
            (root / "stream_manifest.json").write_text(json.dumps(manifest))
            store = module.CheckpointStore(root)
            for data, error in ((payload, None), (payload[:-2], EOFError), (b"\0" * 8, ValueError)):
                response = io.BytesIO(data)
                response.headers = {"Content-Length": "8"}
                with patch.object(module, "urlopen", return_value=response):
                    if error:
                        with self.assertRaises(error):
                            store.get_tensor("weight")
                    else:
                        torch.testing.assert_close(store.get_tensor("weight"), torch.arange(4).half())
            response = io.BytesIO(payload)
            response.headers = {"Content-Length": "7"}
            with patch.object(module, "urlopen", return_value=response):
                with self.assertRaisesRegex(ValueError, "length mismatch"):
                    store.get_tensor("weight")
            (root / "config.json").write_text('{"modified": true}')
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                module.CheckpointStore(root)
            manifest["checkpoint_revision"] = "wrong"
            (root / "stream_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "wrong checkpoint revision"):
                module.CheckpointStore(root)

    def test_truncated_checkpoint_is_rejected_before_loading(self):
        config = {
            "architectures": ["Qwen2_5_VLForConditionalGenerationForJanusVLN"],
            "hidden_size": 3584, "num_hidden_layers": 28, "num_attention_heads": 28,
            "num_key_value_heads": 4, "intermediate_size": 18944, "vocab_size": 152064,
            "reference_frame": "first", "use_vggt_feature": True, "use_sliding_window": False,
            "rope_theta": 1e6, "rms_norm_eps": 1e-6, "hidden_act": "silu", "attention_dropout": 0.,
            "rope_scaling": {"mrope_section": [16, 24, 24]},
            "vision_config": {"depth": 32, "hidden_size": 1280, "num_heads": 16,
                "intermediate_size": 3420, "patch_size": 14, "temporal_patch_size": 2,
                "spatial_merge_size": 2, "window_size": 112, "fullatt_block_indexes": [7, 15, 23, 31]},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config))
            (root / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {"model.norm.weight": "part.safetensors"}}))
            header = json.dumps({"model.norm.weight": {"dtype": "F16", "shape": [4],
                                                       "data_offsets": [0, 8]}}).encode()
            (root / "part.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
            with self.assertRaisesRegex(ValueError, "invalid tensor size"):
                load("memory").checkpoint_bytes(root)

    def test_global_softmax_across_uneven_chunks(self):
        torch.manual_seed(7)
        q, k, v = (torch.randn(1, 2, n, 8) for n in (19, 43, 43))
        # Large, unequal block offsets exercise the online maximum correction.
        k[..., 21:, :] *= 20
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        actual = load("attention").online_attention(q, k, v, query_chunk=7, key_chunk=11)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_chunk_validation(self):
        q = torch.zeros(1, 1, 2, 8)
        with self.assertRaises(ValueError):
            load("attention").online_attention(q, q, q, key_chunk=0)

    def test_raw_history_and_post_attention_crop(self):
        attention = load("attention")
        original = torch.nn.Module()
        original.num_heads, original.head_dim = 16, 64
        original.qkv = torch.nn.Linear(1024, 3072).half()
        original.proj = torch.nn.Linear(1024, 1024).half()
        original.q_norm = torch.nn.LayerNorm(64)
        original.k_norm = torch.nn.LayerNorm(64)
        original.rope = None
        original.eval()
        module = attention.HostAttention(original)
        x = torch.randn(1, 2, 1024).half()
        old = tuple(torch.randn(1, 16, 56, 2, 64).half() for _ in range(2))
        saved = tuple(t.clone() for t in old)
        q, k, v = original.qkv(x).reshape(1, 2, 3, 16, 64).permute(2, 0, 3, 1, 4).unbind(0)
        full = (torch.cat((old[0], k.unsqueeze(2)), 2), torch.cat((old[1], v.unsqueeze(2)), 2))
        expected = torch.nn.functional.scaled_dot_product_attention(
            original.q_norm(q.float()), original.k_norm(full[0].flatten(2, 3).float()),
            full[1].flatten(2, 3).float())
        expected = original.proj(expected.transpose(1, 2).reshape(1, 2, 1024).half())
        actual, cache = module(x, past_key_values=old, use_cache=True)
        torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.001)
        for i in range(2):
            torch.testing.assert_close(old[i], saved[i], atol=0, rtol=0)
            torch.testing.assert_close(cache[i], torch.cat((full[i][:, :, :8], full[i][:, :, -48:]), 2), atol=0, rtol=0)

    def test_memory_and_tmpfs_double_count(self):
        plan = load("memory").memory_plan
        local = plan({"model": 100}, 20, checkpoint_on_tmpfs=True)
        disk = plan({"model": 100}, 20, checkpoint_on_tmpfs=False)
        self.assertEqual(local["conservative_host_peak_bytes"] - disk["conservative_host_peak_bytes"], 100)
        self.assertEqual(local["spatial_tokens_at_step_57"], 92055)
        self.assertEqual(local["allocations_bytes"]["spatial_kv_56_frames"], 8890613760)
        with self.assertRaises(ValueError):
            plan({}, 0, text_capacity=4097)
        admit = load("memory").admit_memory
        with self.assertRaises(MemoryError):
            admit({"model": 100}, 20, {"MemAvailable": 0, "CmaFree": 10**12})
        with self.assertRaises(MemoryError):
            admit({"model": 100}, 20, {"MemAvailable": 10**12, "CmaFree": 0})

    def test_policy_wiring_and_failed_step_guard(self):
        policy_module = load("policy")
        calls = []
        cache = types.SimpleNamespace(max_seq_len=4096, reset_to_position=lambda n: None)
        def aggregate(images, **kwargs):
            calls.append("spatial")
            self.assertEqual(tuple(images.shape), (1, 1, 3, 490, 644))
            tokens = torch.zeros(1, 1, 1615, 2048).half()
            return [tokens, tokens], 5, kwargs["past_key_values"]
        def merge(features):
            self.assertEqual(tuple(features.shape), (1, 17, 23, 2, 2, 2048))
            return torch.ones(391, 3584).half()
        def forward(embeds, positions, logical_len):
            self.assertIn(logical_len, (392, 393))
            if logical_len == 393:
                self.assertIn(400, decoder._prefill_pos_cache)
                self.assertIn(400, decoder._prefill_rope_cache)
            decoder._prefill_pos_cache[400] = positions
            decoder._prefill_rope_cache[400] = positions
            decoder._prefill_pos_cache[384] = positions
            decoder._prefill_rope_cache[384] = positions
            torch.testing.assert_close(embeds[0, :391], torch.full((391, 3584), 2.2).half())
            return embeds
        head_calls = []
        def head(hidden):
            logits = torch.zeros(1, 1, 152064)
            logits[..., 10 if not head_calls else 151645] = 1
            head_calls.append(True)
            return logits
        frontend = types.SimpleNamespace(
            vggt=types.SimpleNamespace(aggregator=aggregate), merger=merge,
            visual=lambda patches, grid_thw: torch.full((391, 3584), 2.).half(),
            model=types.SimpleNamespace(embed_tokens=lambda ids: torch.zeros(*ids.shape, 3584).half()),
            get_rope_index=lambda ids, image_grid_thw: (torch.arange(ids.shape[1]).expand(3, 1, -1), None),
            lm_head=head)
        decoder = types.SimpleNamespace(cache=cache, prefill_execution_plan=lambda n: (400, 64),
            forward_embeds=forward, close=lambda: calls.append("close"),
            _graph_cache=types.SimpleNamespace(clear=lambda: calls.append("graph_clear"),
                                              cache_invariant_ok=lambda: True),
            _prefill_pos_cache={}, _prefill_rope_cache={})
        policy = policy_module.JanusPolicy(frontend, decoder)
        request = dict(input_ids=torch.tensor([[151655] * 391 + [10]]),
                       pixel_values=torch.zeros(1564, 1176).half(),
                       image_grid_thw=torch.tensor([[1, 34, 46]]),
                       images_vggt=[torch.zeros(1, 3, 490, 644).half()])
        result = policy.generate(**request)
        self.assertEqual(result[0, -1].item(), 151645)
        self.assertEqual(calls, ["spatial", "graph_clear"])
        self.assertEqual(result[0, -2:].tolist(), [10, 151645])
        self.assertEqual(set(decoder._prefill_pos_cache), {400})
        self.assertEqual(set(decoder._prefill_rope_cache), {400})
        policy.reset()
        def fail(hidden):
            return torch.full((1, 1, 152064), torch.nan)
        frontend.lm_head = fail
        with self.assertRaisesRegex(RuntimeError, "non-finite logits"):
            policy.generate(**request)
        with self.assertRaises(RuntimeError):
            policy.reset()
        policy.close()
        self.assertEqual(calls[-1], "close")

    def test_close_waits_for_active_generate(self):
        policy_module = load("policy")
        active, release, closing, closed = (threading.Event() for _ in range(4))
        decoder = types.SimpleNamespace(close=closed.set)
        policy = policy_module.JanusPolicy(None, decoder)
        def blocked_generate(**kwargs):
            active.set()
            if not release.wait(5):
                raise RuntimeError("test synchronization timeout")
        policy._generate = blocked_generate
        worker = threading.Thread(target=policy.generate)
        def close():
            closing.set()
            policy.close()
        closer = threading.Thread(target=close)
        worker.start()
        try:
            self.assertTrue(active.wait(2))
            closer.start()
            self.assertTrue(closing.wait(2))
            self.assertFalse(closed.wait(0.05))
        finally:
            release.set()
            worker.join(2)
            if closer.ident is not None:
                closer.join(2)
        self.assertTrue(closed.is_set())
        self.assertFalse(worker.is_alive() or closer.is_alive())


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
