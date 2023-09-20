import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from rethink.dataset import create_distil_dataloader
import rethink.passt.model as passt


class MelSTFT(nn.Module):
    def __init__(
        self,
        n_mels=128,
        sr=32000,
        win_length=800,
        hopsize=320,
        n_fft=1024,
        fmin=0.0,
        fmax=None,
        fmin_aug_range=10,
        fmax_aug_range=2000,
    ):
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.hopsize = hopsize
        self.register_buffer(
            "window", torch.hann_window(win_length, periodic=False), persistent=False
        )
        assert (
            fmin_aug_range >= 1
        ), f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert (
            fmax_aug_range >= 1
        ), f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range
        self.register_buffer(
            "preemphasis_coefficient", torch.as_tensor([[[-0.97, 1]]]), persistent=False
        )

    def forward(self, x):
        x = F.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hopsize,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window,
            return_complex=False,
        )
        x = (x**2).sum(dim=-1)  # power mag
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = (
            self.fmax
            + self.fmax_aug_range // 2
            - torch.randint(self.fmax_aug_range, (1,)).item()
        )

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels,
            self.n_fft,
            self.sr,
            fmin,
            fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = torch.as_tensor(
            F.pad(mel_basis, (0, 1), mode="constant", value=0), device=x.device
        )

        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        melspec = (melspec + 4.5) / 5.0  # fast normalization
        return melspec


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate teacher logits",
    )
    parser.add_argument(
        "--ground_truth_csv",
        default="/media/magalhaes/DCASE2017/schreder_distil_testing_set.csv",
        help="csv file with DCASE2017 labels",
    )
    parser.add_argument(
        "--sounds_dir",
        default="/media/magalhaes/DCASE2017/training_set",
        help="directory with DCASE2017 audios",
    )
    parser.add_argument(
        "--names",
        default="../data/dcase2017.names",
        help="file with DCASE2017 class names",
    )
    parser.add_argument(
        "--audioset-names",
        default="../data/audioset.names",
        help="file with DCASE2017 class names",
    )
    parser.add_argument(
        "--sample_rate", default=32000, type=int, help="sample rate of audio"
    )
    parser.add_argument("--duration", default=10, type=int, help="audipclips duration")
    parser.add_argument("--n_mels", default=128, type=int, help="stfft num bins")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument(
        "--num_workers", default=8, type=int, help="num workers for dataloader"
    )
    parser.add_argument(
        "--conf-thres", default=0.5, type=float, help="confidence threshold"
    )
    parser.add_argument("--device", default="cuda:0", type=str, help="device")

    args = parser.parse_args()

    names = []
    for line in open(args.names).readlines():
        names.append(line.strip())

    audioset_2_dcase_index_mask = []
    audioset_names_df = pd.read_csv(args.audioset_names)
    for name in names:
        idx = audioset_names_df[audioset_names_df["mid"] == name]["index"].values
        audioset_2_dcase_index_mask.append(idx[0])

    print("Output index mask:")
    print(audioset_2_dcase_index_mask)

    loader = create_distil_dataloader(
        args.ground_truth_csv,
        None,
        names,
        args.sample_rate,
        args.duration,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = passt.get_ensemble_model(
        [
            ("passt_s_swa_p16_128_ap476", 10, 10),
            ("passt_s_swa_p16_s14_128_ap471", 14, 14),
            ("passt_s_swa_p16_s12_128_ap473", 12, 12),
        ]
    )
    model.to(device)
    model.eval()

    mel_extractor = MelSTFT(n_mels=args.n_mels, sr=args.sample_rate)
    mel_extractor.to(device)

    predictions = []
    targets = []
    total_audioclips = 0
    with torch.no_grad():
        for i, (waveform, y) in enumerate(loader):
            waveform = waveform.to(device)
            # N, L
            spectrogram = mel_extractor(waveform)[:, :, :998]
            # N, BINS, WIDTH
            spectrogram = spectrogram.unsqueeze(1)
            # N, 1, BINS, WIDTH
            ensemble_logits, _ = model(spectrogram)
            ensemble_logits = ensemble_logits[:, audioset_2_dcase_index_mask].cpu()

            outputs = torch.sigmoid(ensemble_logits)
            outputs = outputs.argmax(1)

            predictions.append(outputs.numpy())
            targets.append(y.numpy().argmax(1))
            total_audioclips += y.shape[0]

        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        f1 = f1_score(predictions, targets, average="macro", zero_division=0)
        precision = precision_score(
            predictions, targets, average="macro", zero_division=0
        )
        recall = recall_score(predictions, targets, average="macro", zero_division=0)

        confusion_matrix_res = confusion_matrix(predictions, targets)

        print(f"F1 score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(predictions)
        print(targets)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix_res,
            display_labels=[
                "Train horn",
                "Train",
                "Air horn",
                "Bus",
                "Ambulance (siren)",
                "Civil defense siren",
                "Fire engine, fire truck (siren)",
                "Truck",
                "Car alarm",
                "Car",
                "Police car (siren)",
                "Car passing by",
                "Reversing beeps",
                "Bicycle",
                "Skateboard",
                "Screaming",
                "Motorcycle",
            ],
        )
        disp.plot()
        disp.figure_.savefig("./confusion_matrix.png")


if __name__ == "__main__":
    main()
