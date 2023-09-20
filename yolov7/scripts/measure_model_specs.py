import time
import argparse

import numpy as np
import torch

from thop import profile

from yolov7.models.yolo import Model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="measure_model_specs.py")
    parser.add_argument(
        "--cfg-path",
        default="../cfg/deploy/yolov7.yaml",
        type=str,
        help="path to model config file",
    )
    parser.add_argument(
        "--num-classes", type=int, default=8, help="Number of classes in the dataset"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="size of each image dimension"
    )
    parser.add_argument("--device", type=str, default="cpu", help="device to use")
    args = parser.parse_args()

    image_size = (args.img_size, args.img_size)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Create model
    model = Model(
        args.cfg_path,
        ch=3,
        nc=args.num_classes,
        anchors=3,
    ).to(device)

    # Dummy input
    input = torch.zeros((3,) + (image_size)).unsqueeze(0)

    model.to(device)
    input = input.to(device)

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
