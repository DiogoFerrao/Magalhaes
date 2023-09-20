import os
import argparse
import shutil

import torch

from tqdm import tqdm

from rethink.models import create_model
from rethink.preprocess import Loader, Trimmer, Padder, LogMelSpectrogramExtractorModel


def save_prediction(out_dir, filename, y_hat):
    with open(os.path.join(out_dir, filename + ".txt"), "w") as f:
        for i in range(len(y_hat)):
            f.write(str(i) + " " + str(y_hat[i].item()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate detections from a trained model"
    )
    parser.add_argument("input_dir", type=str, help="Input directory with wav files")
    parser.add_argument(
        "--model",
        default="yolov7_tiny",
        type=str,
        help="Model name. Options: (yolo, yolov7, yolov7_tiny)",
    )
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--num_classes", default=7, type=int)
    parser.add_argument("--out_dir", default="/media/magalhaes/sound/outputs", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--names", default="./data/schreder.names", type=str)

    parser.add_argument("--sample_rate", default=22050, type=int)
    parser.add_argument("--n_mels", default=128, type=int)
    parser.add_argument("--image_length", default=256, type=int)
    parser.add_argument("--duration", default=10, type=int)
    args = parser.parse_args()

    # Create the output directory if it does not exist
    os.makedirs(args.out_dir, exist_ok=True)

    # Copy the names file to the output directory using shutil
    shutil.copy(args.names, args.out_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = create_model(args.model, args.num_classes, args.checkpoint, args.device)
    model.to(device)
    model.eval()

    mel_extractor = LogMelSpectrogramExtractorModel(
        args.sample_rate,
        args.n_mels,
        args.image_length,
        args.duration,
    )
    mel_extractor.to(device)

    loader = Loader(args.sample_rate, device=args.device)
    trimmer = Trimmer(args.duration, args.sample_rate)
    padder = Padder(args.duration, args.sample_rate)

    paths = os.listdir(args.input_dir)

    with tqdm(total=len(paths)) as t:
        for path in paths:
            if not path.endswith(".wav"):
                continue
            filename = os.path.basename(path).split(".")[0]
            t.set_postfix(path=path)
            t.update()

            x = loader(os.path.join(args.input_dir, path))
            x = trimmer(x)
            x = padder(x)
            x = mel_extractor(x)
            # add batch dimension
            x = x.unsqueeze(0)

            y_hat = model(x)[0]
            y_hat = torch.sigmoid(y_hat)

            save_prediction(args.out_dir, filename, y_hat)
