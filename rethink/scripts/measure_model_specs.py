import time
import argparse

import numpy as np
import torch

from thop import profile
from rethink.models import create_model


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--arch", default="yolov7_tiny", type=str, help="device")
    parser.add_argument("--shape", nargs="+", type=int, default=(3, 128, 256))

    args = parser.parse_args()

    # Create model
    model = create_model(args.arch, 7)

    # Dummy input
    input = torch.zeros(args.shape).unsqueeze(0).to("cuda:0")

    # Measure Flops
    flops = profile(model, inputs=(input,), verbose=False)[0] / 1e9 * 2
    print("%.9f GFLOPS" % (flops))
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(
        x.numel() for x in model.parameters() if x.requires_grad
    )  # number gradients

    print(f"Number of parameters {n_p}")
    print(f"Number of gradients {n_g}")

    # Measure memory usage in Bytes
    mem_params = sum(
        [param.nelement() * param.element_size() for param in model.parameters()]
    )
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs  # in bytes
    print(f"Memory usage: {mem/1e6} MB")

    inner_times = []
    start_time = time_synchronized()
    for i in range(5000):
        inner_start_time = time_synchronized()
        model(input)
        inner_times.append(time_synchronized() - inner_start_time)
    total_time = time_synchronized() - start_time

    print(f"FPS: {1.0 / np.mean(inner_times)}")
    print(f"Time per inference (ms): {np.mean(inner_times)*1000}")
    print(f"Total Time: {total_time}")
