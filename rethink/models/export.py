import argparse

import torch

from rethink.models import create_model
from rethink.preprocess import LogMelSpectrogramExtractorModel


class DeploymentModel(torch.nn.Module):
    def __init__(self, extractor, model):
        super().__init__()
        self.extractor = extractor
        self.model = model

    def forward(self, x):
        x = self.extractor(x)
        return torch.sigmoid(self.model(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a model to ONNX format for deployment"
    )
    parser.add_argument("weights", type=str, help="Path to model weights")
    parser.add_argument(
        "--arch", type=str, default="yolov7_tiny", help="Model architecture"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=22050, help="Sample rate of audios"
    )
    parser.add_argument(
        "--duration", type=int, default=10, help="Audio duration in seconds"
    )
    parser.add_argument("--n_mels", type=int, default=128, help="Number of mel bins")
    parser.add_argument(
        "--spectrogram_length", type=int, default=256, help="Width of spectrogram"
    )
    parser.add_argument(
        "--num_classes",
        nargs="+",
        type=int,
        default=7,
        help="Number of classes detected by the model",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size used during inference"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for model"
    )
    opt = parser.parse_args()
    print(opt)

    # Set Device
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    # Input
    waveform = torch.zeros((opt.batch_size, opt.sample_rate * opt.duration)).to(device)

    # Load model
    model = create_model(opt.arch, opt.num_classes, opt.weights, device="cpu")
    extractor = LogMelSpectrogramExtractorModel(
        opt.sample_rate, opt.n_mels, opt.spectrogram_length, opt.duration, export=True
    )
    complete_model = DeploymentModel(extractor, model)
    complete_model.float()  # to FP32
    complete_model.eval()
    complete_model.to(device)
    y = complete_model(waveform)  # dry run

    try:
        import onnx

        print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
        export_path = opt.weights.rstrip(".pth") + ".onnx"  # filename
        torch.onnx.export(
            complete_model,
            waveform,
            export_path,
            verbose=True,
            opset_version=11,
            input_names=["waveform"],
            output_names=["classes"],
        )

        # Checks
        onnx_model = onnx.load(export_path)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(
            onnx.helper.printable_graph(onnx_model.graph)
        )  # print a human readable model
        print(f"ONNX export success, saved as {export_path}")
    except Exception as e:
        print("ONNX export failure: %s" % e)

    # Finish
    print("\nExport complete. Visualize with https://github.com/lutzroeder/netron.")
