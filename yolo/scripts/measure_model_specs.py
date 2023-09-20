import time
import argparse

import numpy as np
import torch

from thop import profile

from yolo.models.models import Darknet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="measure_model_specs.py")
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="../models/yolov4-csp.cfg",
        help="path to model config file",
    )
    parser.add_argument("--device", type=str, default="cpu", help="device to use")
    args = parser.parse_args()

    image_size = (640, 640)

    # Create model
    model = Darknet(args.cfg_path, image_size)

    # Dummy input
    input = torch.zeros((3,) + (image_size)).unsqueeze(0)

    # Measure Flops
    flops = profile(model, inputs=(input,), verbose=False)[0] / 1e9 * 2
    print("%.9f GFLOPS" % (flops))
    n_parameters = sum(x.numel() for x in model.parameters())
    n_gradients = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"Number of parameters: {n_parameters}")
    print(f"Number of gradients: {n_gradients}")

    # Measure memory usage in Bytes
    mem_params = sum(
        [param.nelement() * param.element_size() for param in model.parameters()]
    )
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs  # in bytes
    print(f"Memory usage: {mem/1e6} MB")

    model.to(args.device)
    input = input.to(args.device)

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
