"""Explicit R1 check; run under the device lock, outside automatic host tests."""
import gc
import time

import torch
import rpu_backend

from rpu_backend.adapters.janus_vln.attention import RPUProjection, rpu_online_attention


def main():
    torch.manual_seed(17)
    torch.set_num_threads(4)
    torch.rpu.set_caching_allocator(True)
    with torch.inference_mode():
        for m, k, n in ((65, 588, 1024), (65, 1176, 1280), (65, 1280, 3420),
                        (65, 3420, 1280), (65, 8192, 4096), (1, 3584, 152064)):
            linear = torch.nn.Linear(k, n).half().eval()
            x = torch.randn(m, k).half()
            expected = torch.nn.functional.linear(x.float(), linear.weight.float(), linear.bias.float())
            projection = RPUProjection(linear)
            assert projection.linear.weight.device.type == "rpu"
            for repeat in range(2):
                start = time.perf_counter()
                actual = projection(x)
                error = (actual.float() - expected).norm() / expected.norm()
                print("projection", (m, k, n), repeat, time.perf_counter() - start, float(error), flush=True)
                assert actual.shape == expected.shape and actual.isfinite().all() and error < 0.1
            del projection, linear, expected, actual, x
            gc.collect()
        for rows, keys, width in ((19, 43, 64), (1615, 1615, 64), (1615, 3230, 64), (64, 64, 80), (19, 1100, 64), (1, 92055, 64)):
            q, k, v = (torch.randn(1, 16, size, width) for size in (rows, keys, keys))
            if keys == 1100:
                q.fill_(1)
                k.fill_(1)
                k[:, :, 1024:] = -100
            expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            start = time.perf_counter()
            actual = rpu_online_attention(q, k, v)
            error = (actual - expected).norm() / expected.norm()
            print("attention", (rows, keys, width), time.perf_counter() - start, float(error), flush=True)
            assert actual.shape == expected.shape and actual.isfinite().all() and error < 0.05


if __name__ == "__main__":
    main()
