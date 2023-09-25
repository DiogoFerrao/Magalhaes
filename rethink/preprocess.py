"""
1- load a file
2- pad the signal
3- extract the log spectrogram
4- normalize the spectrogram
5- save the spectrogram
"""
import os
from typing import Optional
import math
import argparse

import numpy as np
import pickle as pkl
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision

from rethink.stft.torchlibrosa import LibrosaMelSpectrogram
from rethink.stft.effi import MelSTFT


class Loader:
    """Loader is responsible for loading an audio file"""

    def __init__(self, sample_rate, mono=True, device="cpu"):
        self.sample_rate = sample_rate
        self.mono = mono
        self.device = device

    def __call__(self, path):
        audio_time_series, source_sr = torchaudio.load(path)
        audio_time_series = torch.mean(audio_time_series, dim=0)
        audio_time_series = torchaudio.functional.resample(
            audio_time_series, orig_freq=source_sr, new_freq=self.sample_rate
        )

        return audio_time_series.to(self.device)


class Trimmer:
    def __init__(self, sample_rate: int, duration: int):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = self.sample_rate * self.duration

    def __call__(self, audio_time_series):
        # Keep 10 seconds from the middle
        if self.duration and len(audio_time_series) > self.n_samples:
            n_excess_samples = len(audio_time_series) - self.n_samples
            audio_time_series = audio_time_series[
                math.floor(n_excess_samples / 2) : len(audio_time_series)
                - math.ceil(n_excess_samples / 2)
            ]
        return audio_time_series


class Padder:
    """Padder is responsible for padding the signal"""

    def __init__(self, sample_rate: int, duration: Optional[int] = None):
        self.duration = duration
        self.sample_rate = sample_rate

    def __call__(self, signal):
        if self.duration is not None:
            num_missing_items = int(self.sample_rate * self.duration) - len(signal)
            if num_missing_items > 0:
                return F.pad(
                    signal,
                    (
                        math.ceil(num_missing_items / 2),
                        math.floor(num_missing_items / 2),
                    ),
                )
        return signal


class LogMelSpectrogramExtractorModel(nn.Module):
    """LogMelSpectrogramExtractorModel is responsible for extracting the log spectrogram

    Args:
        sample_rate (int): sample rate of the audio
        n_mels (int): number of mel filterbanks
        length (int): width of the spectrogram
        duration (int): duration of the audio
        hop_sizes (list): hop sizes for the spectrogram
        window_sizes (list): window sizes for the spectrogram
        export (bool): if True, the extractor will use the mel spectrogram from the
        torchlibrosa library, which can be exported to ONNX. Otherwise, the extractor
        will use the mel spectrogram from torchaudio.
    """

    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        length: int,
        duration: int,
        hop_sizes=[10, 25, 50],
        window_sizes=[25, 50, 100],
        export=False,
    ):
        super().__init__()
        self.num_channels = 3
        self.sample_rate = sample_rate
        self.hop_sizes = hop_sizes
        self.window_sizes = window_sizes
        self.n_mels = n_mels
        self.length = length
        self.duration = duration
        self.export = export

        assert self.num_channels == len(self.hop_sizes) and self.num_channels == len(
            self.window_sizes
        )

        self.resize_transform = torchvision.transforms.Resize(
            (self.n_mels, self.length), antialias=None
        )

        n_fft = int(self.sample_rate / self.duration)
        self.mel_spec_transforms = []
        for i in range(3):
            window_length = int(round(self.window_sizes[i] * self.sample_rate / 1000))
            hop_length = int(round(self.hop_sizes[i] * self.sample_rate / 1000))
            if self.export:
                # mel_spec = ConvertibleSpectrogram(
                #     sr=self.sample_rate,
                #     n_fft=n_fft,
                #     hop_size=hop_length,
                #     win_size=window_length,
                #     n_mel=self.n_mels,
                # )
                # mel_spec.set_mode("DFT", "store")
                mel_spec = LibrosaMelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=window_length,
                    n_mels=self.n_mels,
                )
            else:
                mel_spec = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=n_fft,
                    win_length=window_length,
                    hop_length=hop_length,
                    n_mels=self.n_mels,
                )
            self.mel_spec_transforms.append(mel_spec)
        self.mel_spec_transforms = nn.ModuleList(self.mel_spec_transforms)

    def forward(self, signal: torch.Tensor):
        no_batch = signal.dim() == 1
        if no_batch:
            signal = signal.unsqueeze(0)

        N = signal.size(0)
        spectgrms = torch.zeros(
            (self.num_channels, N, self.n_mels, self.length), device=signal.device
        )
        for i in range(self.num_channels):
            # Spectogram
            spectgrm = self.mel_spec_transforms[i](signal)
            # Log spectogram
            eps = 1e-6
            log_spec = torch.log(spectgrm + eps)
            if self.export:
                log_spec = log_spec.unsqueeze(0)
                log_spec = self.resize_transform(log_spec)[0]
            else:
                log_spec = self.resize_transform(log_spec)

            log_spec = (log_spec + 4.5) / 5.0  # fast normalization
            spectgrms[i] = log_spec

        spectgrms = spectgrms.permute(1, 0, 2, 3)
        if no_batch:
            spectgrms = spectgrms.squeeze(0)
        return spectgrms


class PitchShift:
    def __init__(self, sample_rate: int, device="cpu"):
        self.sample_rate = sample_rate
        self.device = device

    def __call__(self, waveform):
        pitch_shift = torch.randint(-2, 2 + 1, (1,)).item()
        return (
            torchaudio.functional.pitch_shift(
                waveform, sample_rate=self.sample_rate, n_steps=pitch_shift
            )
            .to(self.device)
            .squeeze(0)
        )


class MinMaxNormalizer:
    """MinMaxNormalizer is responsible for normalizing the spectrogram"""

    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def __call__(self, spectgrms):
        for spectgrm in spectgrms:
            norm_spectrogram = (spectgrm - spectgrm.min()) / (
                spectgrm.max() - spectgrm.min()
            )
            spectgrm = norm_spectrogram * (self.max - self.min) + self.min
        return spectgrms


class Packager:
    """Packager is responsible for packaging the information"""

    def __init__(self, save_waveform=False, save_spectogram=True):
        self.save_waveform = save_waveform
        self.save_spectogram = save_spectogram

    def package(self, signal, spectgrms, labels, dataset):
        entry = {}
        if self.save_waveform:
            entry["waveform"] = signal.cpu().numpy()
        if self.save_spectogram:
            entry["spectograms"] = np.array(spectgrms)
        entry["dataset"] = dataset
        entry["target"] = labels
        return entry

    def dump(self, output_path, output_name, data):
        with open(os.path.join(output_path, output_name), "wb") as handler:
            pkl.dump(data, handler, protocol=pkl.HIGHEST_PROTOCOL)


class Preprocessor:
    """Preprocessor is responsible for preprocessing the audio files"""

    def __init__(self, loader, trimmer, padder, extractor, packager):
        self.loader = loader
        self.trimmer = trimmer
        self.padder = padder
        self.extractor = extractor
        self.packager = packager

    def preprocess(self, path, labels, dataset):
        signal = self.loader(path)
        signal = self.trimmer(signal)
        signal = self.padder(signal)
        spectgrms = self.extractor(signal).cpu()
        return self.packager.package(signal, spectgrms, labels, dataset)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files into 3 channel spectrograms and stores the result in an npy file"
    )
    parser.add_argument("csv_file", type=str, help="csv file with the dataset")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate ")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of mel filterbanks which defines the height of the spectrogram")
    parser.add_argument("--image_length", type=int, default=256, help="Width of 3 channel spectrogram after preprocessing")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of audios (larger audios will clipped and smaller will be padded)")
    parser.add_argument("--effi_extractor", action="store_true", help="Use mel spectrogram extractor from Effi")
    parser.add_argument("--export_extractor", action="store_true", help="Use mel spectrogram extractor from torchlibrosa that is compatible with ONNX")
    parser.add_argument("--waveform_only", action="store_true", help="Only save waveform")
    parser.add_argument("--labels_path", type=str, default="./data/schreder.names", help="Path to text file with label names")
    parser.add_argument(
        "--output_dir", type=str, default="/media/magalhaes/sound/spectograms", help="Path to output directory"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    # fmt: on

    audios_df = pd.read_csv(args.csv_file, skipinitialspace=True)

    dataset_name = os.path.basename(args.csv_file).split(".")[0]

    os.makedirs(args.output_dir, exist_ok=True)

    label_to_id = {}
    for i, label in enumerate(open(args.labels_path).readlines()):
        label_to_id[label.strip()] = i

    # Get number of splits and
    splits_arr = list(audios_df["split"].unique())
    audio_splits = [audios_df.loc[audios_df["split"] == i] for i in splits_arr]

    loader = Loader(sample_rate=args.sample_rate, device=args.device)
    trimmer = Trimmer(args.sample_rate, args.duration)
    padder = Padder(sample_rate=args.sample_rate, duration=args.duration)

    if args.effi_extractor:
        extractor = MelSTFT(n_mels=args.n_mels, sr=args.sample_rate).to(args.device)
    else:
        extractor = LogMelSpectrogramExtractorModel(
            args.sample_rate,
            args.n_mels,
            args.image_length,
            args.duration,
            export=args.export_extractor,
        ).to(args.device)

    packager = Packager(
        save_waveform=args.waveform_only, save_spectogram=not args.waveform_only
    )

    preprocessor = Preprocessor(loader, trimmer, padder, extractor, packager)

    for j, dataset_split in enumerate(audio_splits):
        values = []
        split_filename = f"{dataset_name}_{splits_arr[j]}.pkl"
        print(f"Processing {split_filename}")
        with tqdm(total=len(dataset_split)) as pbar:
            for i, row in dataset_split.iterrows():
                labels = [
                    int(row[key])
                    for key in audios_df.columns
                    if key in label_to_id.keys()
                ]
                entry = preprocessor.preprocess(
                    row["audio_filename"], labels, row["dataset"]
                )
                values.append(entry)
                pbar.update(1)
        preprocessor.packager.dump(args.output_dir, split_filename, values)
